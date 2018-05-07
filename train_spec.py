import scipy.misc
import time
import os
import timeit
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import pprint
import librosa
from tensorflow.python.ops import control_flow_ops

class Rating_Prediction(object):

  def __init__(self, 
               sess,
               batch_size=64,
               epoch=50,
               checkpoint_dir=None):

    self.sess = sess
    self.batch_size = batch_size
    self.epoch = epoch
    self.tfrecords = './tfrecords/train_spec_50_100.tfrecords'
    self.tfrecords_val = './tfrecords/valid_spec_50_100.tfrecords'
    self.checkpoint_dir = checkpoint_dir
    self.keep_prob = 1.
    self.keep_prob_var = 1.
    self.save_freq = 1000
    self.lr = 0.00002


    self.build_model()

  def parser(self, record):
      keys_to_features = {
          'X_audio': tf.FixedLenFeature([], tf.string),
           'Y_i': tf.FixedLenFeature([], tf.string),
           'sid': tf.FixedLenFeature([], tf.string),
           # 'is_train': tf.FixedLenFeature([], tf.string),
           
      }
      parsed = tf.parse_single_example(record, keys_to_features)

      audio = tf.decode_raw(parsed['X_audio'], tf.float32)
      Y_rat = tf.decode_raw(parsed['Y_i'], tf.float32)
      sid = tf.decode_raw(parsed['sid'], tf.float32)
      # audio = tf.reshape(audio, [128, 512])
      # Y_rat = tf.reshape(Y_rat, [50])

      # spec.set_shape([128,646])
      # lyrics_embed.set_shape([300])
      # Y_rat.set_shape([50])
      # sid.set_shape([1])

      return audio, Y_rat, sid

  def average_gradients(self, tower_grads):
    """ Calculate the average gradient for each shared variable across towers.
    Note that this function provides a sync point across al towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
        list is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # each grad is ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dim to gradients to represent tower
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension that we will average over below
            grads.append(expanded_g)

        # Build the tensor and average along tower dimension
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # The Variables are redundant because they are shared across towers
        # just return first tower's pointer to the Variable
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

    pass

  def build_model(self):

    def init_weights(shape):
      return tf.Variable(tf.random_normal(shape, stddev=.1))

    def init_biases(shape):
      return tf.Variable(tf.zeros(shape))

    dataset = tf.data.TFRecordDataset(self.tfrecords)
    dataset = dataset.map(self.parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(self.epoch)
    dataset = dataset.batch(self.batch_size)
    iterator = dataset.make_one_shot_iterator()
    spect, ratt, sidt = iterator.get_next()

    datasetv = tf.data.TFRecordDataset(self.tfrecords_val)
    datasetv = datasetv.map(self.parser)
    # datasetv = datasetv.shuffle(buffer_size=10000)
    datasetv = datasetv.repeat(self.epoch*1000000)
    datasetv = datasetv.batch(500)
    iteratorv = datasetv.make_one_shot_iterator()
    specv, ratv, sidv = iteratorv.get_next()

    self.is_valid = tf.placeholder(dtype=bool, shape=())
    self.keep_prob_var_valid = tf.Variable(self.keep_prob, trainable=False)
    self.keep_prob = 0.5
    self.keep_prob_var_train = tf.Variable(self.keep_prob, trainable=False)
    bs, spec, rat, sid, self.keep_prob_var = tf.cond(self.is_valid, lambda: [500, specv, ratv, sidv, self.keep_prob_var_valid], 
                                                                        lambda: [self.batch_size, spect, ratt, sidt, self.keep_prob_var_train])
    # spec, lyrics, rat, sid = spect, lyricst, ratt, sidt

    self._spec = spec
    self.spec = tf.reshape(spec, [bs, 128, 512, 1])
    # self.spec = tf.expand_dims(spec, -1)

    self.rat = tf.reshape(rat, [bs, 50])
    self.sid = sid
    self.bs = bs

    self.pred_only_spec = self.spec_cnn(self.spec)

    self.losses = []
    self.loss = tf.losses.absolute_difference(labels=self.rat, predictions=self.pred_only_spec)
    # self.loss = tf.reduce_mean(tf.abs(tf.subtract(self.pred_only_spec, self.rat)))
    self.losses.append(self.loss)

    opt = tf.train.AdamOptimizer(self.lr, 0.9, 0.999)

    self.all_grads = []
    self.grads = opt.compute_gradients(self.loss)
    self.all_grads.append(self.grads)
    self.avg_grads = self.average_gradients(self.all_grads)
    self.opt = opt.apply_gradients(self.avg_grads)

    self.saver = tf.train.Saver()
  

  def batch_norm(self, x, n_out, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

  def discriminator(self, inputs):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    with tf.variable_scope('D'):
      lay1_D = tf.layers.dense(inputs, 50, activation=None, kernel_regularizer=regularizer)
      lay2_D = tf.layers.dense(lay1_D, 32, activation=None, kernel_regularizer=regularizer)
      lay3_D = tf.layers.dense(lay2_D, 16, activation=None, kernel_regularizer=regularizer)
      out = tf.layers.dense(lay3_D, 1, activation=None, kernel_regularizer=regularizer)
      return out

  def spec_cnn(self, inputs):
    # inputs = tf.reshape(inputs, [self.bs, 128, 512, 1])
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    with tf.variable_scope("only_spec"):

      conv1 = tf.layers.conv2d(inputs, filters=8, kernel_size=(3,7), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                  activation=tf.nn.relu ,kernel_regularizer=regularizer)
      conv1 = tf.layers.dropout(conv1, self.keep_prob_var)

      conv1 = tf.layers.conv2d(conv1, filters=8, kernel_size=(7,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                  activation=tf.nn.relu ,kernel_regularizer=regularizer)
      conv1 = tf.layers.dropout(conv1, self.keep_prob_var)
      
      conv1p = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv1p.shape)

      conv2 = tf.layers.conv2d(conv1p, filters=16, kernel_size=(3,7), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu, kernel_regularizer=regularizer)
      conv2 = tf.layers.dropout(conv2, self.keep_prob_var)
      
      conv2 = tf.layers.conv2d(conv2, filters=16, kernel_size=(7,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu, kernel_regularizer=regularizer)
      conv2 = tf.layers.dropout(conv2, self.keep_prob_var)
      
      conv2p = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv2p.shape)
      
      conv3 = tf.layers.conv2d(conv2p, filters=32, kernel_size=(3,7), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu,kernel_regularizer=regularizer)
      conv3 = tf.layers.dropout(conv3, self.keep_prob_var)
      
      conv3 = tf.layers.conv2d(conv3, filters=32, kernel_size=(7,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu,kernel_regularizer=regularizer)   
      conv3 = tf.layers.dropout(conv3, self.keep_prob_var)
      
      conv3p = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv3p.shape)

      conv4 = tf.layers.conv2d(conv3p, filters=64, kernel_size=(3,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu, kernel_regularizer=regularizer)
      conv4 = tf.layers.dropout(conv4, self.keep_prob_var)

      conv4p = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv4p.shape)

      # conv5 = tf.layers.conv2d(conv4p, filters=256, kernel_size=(3,3), kernel_initializer=tf.contrib.layers.xavier_initializer(),
      #         activation=tf.nn.relu, kernel_regularizer=regularizer)
      # conv5 = tf.layers.dropout(conv5, self.keep_prob_var)
      # conv5p = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      # print (conv5p.shape)

      flat = tf.layers.flatten(conv4p)
      print (flat.shape)
      lay1 = tf.layers.dense(flat, 1024, activation=None, kernel_regularizer=regularizer)
      # lay1 = tf.layers.dropout(lay1, 0.2)
      lay2 = tf.layers.dense(lay1, 1024, activation=None, kernel_regularizer=regularizer)
      # lay2 = tf.layers.dropout(lay2, 0.2)    
      lay3 = tf.layers.dense(lay2, 1024, activation=None, kernel_regularizer=regularizer)
      output = tf.layers.dense(lay3, 50, activation=None)

      return output

  def train(self, sess):
    # tf.initialize_all_variables().run()
    print ('initializing...opt')
    opt = self.opt
    try:
      init = tf.global_variables_initializer()
      sess.run(init)
    except AttributeError:
      init = tf.intializer_all_varialble()
      sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    sample_spec, sample_rat, sample_sid = self.sess.run([self.spec, self.rat, self.sid], feed_dict={self.is_valid:False})
    print ('sample spec shape: ', sample_spec.shape)
    print ('sample rat shape: ', sample_rat.shape)
    counter = 0
    # count of num of samples
    num_examples = 0
    for record in tf.python_io.tf_record_iterator(self.tfrecords):
      num_examples += 1
    print ("total num of patches in tfrecords", self.tfrecords,":  ", num_examples)

    num_batches = num_examples / self.batch_size
    print ('batches per epoch: ', num_batches)
    batch_idx = 0
    current_epoch = 0
    batch_timings = []
    losses = []
    v_losses = []

    try:
      while not coord.should_stop():

        start = timeit.default_timer()       
        _opt, _loss = self.sess.run([opt, self.loss], feed_dict={self.is_valid:False})
        v_loss = self.sess.run(self.loss, feed_dict={self.is_valid:True})
      
        end = timeit.default_timer()
        batch_timings.append(end - start)
        losses.append(_loss)
        v_losses.append(v_loss)


        if counter % 5 == 0:
          print('{}/{} (epoch {}), '
                        'train_mse_loss = {:.5f}, '
                        'valid_mse_loss = {:.5f}, '
                        ' time/batch = {:.5f}, '
                        'mtime/batch = {:.5f}'.format(counter,
                                                      self.epoch * num_batches,
                                                      current_epoch,
                                                      _loss,
                                                      v_loss,
                                                      end - start,
                                                      np.mean(batch_timings)))


        if counter % 100 == 0:
          pred, y = self.sess.run([self.pred_only_spec, self.rat], feed_dict={self.is_valid:False})

          print ('pred',pred)
          print ('y',y)

        batch_idx += 1
        counter += 1
        if (counter) % 2000 == 0 and (counter) > 0:
          self.saver.save(self.sess, "./only_spec"+str(counter)+"_th")
        if batch_idx >= num_batches:
          current_epoch += 1
          #reset batch idx
          batch_idx = 0
        if current_epoch >= self.epoch:
          print (str(self.epoch),': epoch limit')
          print ('saving last model at iteration',str(counter))
          self.saver.save(self.sess, "./only_spec_final")
          break

    except tf.errors.OutOfRangeError:
      print('done training')
      pass
    finally:
      coord.request_stop()
    coord.join(threads)


# flags = tf.app.flags
# flags.DEFINE_integer("epoch", 200, "Number of epochs [100]")
# flags.DEFINE_integer("training_step", 10000, "Number of training steps [10000]")
# flags.DEFINE_integer("batch_size", 16, "The size of batch sizes [100]")
# flags.DEFINE_float("learning_rate", 3e-4, "The learning rate of optimizing algorithm [0.0003]")
# flags.DEFINE_integer("lam", .01, "Lambda regularizer [0.01]")
# flags.DEFINE_string("checkpoint_dir", "checkpoint_conv_vae", "Checkpoint directory [checkpoint_dir]")
# FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):



  with tf.Session() as sess:
    only_audio = Rating_Prediction(sess)
  
    only_audio.train(sess)

if __name__ == '__main__':
  tf.app.run()
  np.save('./spec_vloss.npy', v_losses)
  np.save('./spec_tloss.npy', losses)