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
    self.lr = 0.0001
    self.gan_lambda = 1.
    self.l1_lambda = 1.


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
    
  def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars_dict = {}
    self.g_vars_dict = {}

    for var in t_vars:
      if var.name.startswith('D_'):
        self.d_vars_dict[var.name] = var
      if var.name.startswith('G_'):
        self.g_vars_dict[var.name] = var
    self.d_vars = list(self.d_vars_dict.values())
    self.g_vars = list(self.g_vars_dict.values())

    for x in self.d_vars:
        assert x not in self.g_vars
    for x in self.g_vars:
        assert x not in self.d_vars

    self.all_vars = t_vars

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
    self.GG = []

    dummy_input = self.rat
    dummy = self.discriminator(dummy_input, reuse=0)

    G = self.generator(self.spec, name='G_model')
    G = tf.reshape(G, [bs, 50])
    self.GG.append(G)

    self.real_input = self.rat
    self.fake_input = G
    d_real_logits = self.discriminator(self.real_input, reuse=2)
    d_fake_logits = self.discriminator(self.fake_input, reuse=2)


    self.g_losses = []
    self.g_l1_losses = []
    self.g_adv_losses = []
    self.d_real_losses = []
    self.d_fake_losses = []
    self.d_losses = []

    # wgan
    d_real_loss = -tf.reduce_mean(d_real_logits)
    d_fake_loss = tf.reduce_mean(d_fake_logits)
    g_adv_loss = -tf.reduce_mean(d_fake_logits)

    d_loss = self.gan_lambda * (d_real_loss + d_fake_loss) 

    g_l1_loss = tf.reduce_mean(tf.abs(tf.subtract(G, self.rat)))
    g_loss = self.l1_lambda * g_l1_loss + self.gan_lambda * g_adv_loss 

    epsilon = tf.random_uniform([], 0.0, 1.0)
    x_hat = self.real_input*epsilon + (1-epsilon)*self.fake_input
    d_hat = self.discriminator(x_hat, reuse=2)
    gradients = tf.gradients(d_hat, x_hat)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = 3*tf.reduce_mean((slopes-1.0)**2)
    d_loss = d_loss + gradient_penalty

    self.g_l1_losses.append(g_l1_loss)
    self.g_adv_losses.append(g_adv_loss)
    self.g_losses.append(g_loss)
    self.d_real_losses.append(d_real_loss)
    self.d_fake_losses.append(d_fake_loss)
    self.d_losses.append(d_loss)

    all_d_grads = []
    all_g_grads = []

    d_opt = tf.train.AdamOptimizer(self.lr, 0.9, 0.999)
    g_opt = tf.train.AdamOptimizer(self.lr, 0.9, 0.999)

    self.get_vars()

    g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)
    all_g_grads.append(g_grads)
    avg_g_grads = self.average_gradients(all_g_grads)
    self.g_opt = g_opt.apply_gradients(avg_g_grads)

    d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)
    all_d_grads.append(d_grads)
    avg_d_grads = self.average_gradients(all_d_grads)
    self.d_opt = d_opt.apply_gradients(avg_d_grads)
    
    # self.all_grads = []
    # self.grads = opt.compute_gradients(self.loss)
    # self.all_grads.append(self.grads)
    # self.avg_grads = self.average_gradients(self.all_grads)
    # self.opt = opt.apply_gradients(self.avg_grads)

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

  def discriminator(self, inputs, reuse=0, name='D_model'):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    with tf.variable_scope(name) as scope:
      if reuse == 2:
        scope.reuse_variables()

      inputs = tf.reshape(inputs, [-1,50,1])
      conv1 = tf.layers.conv1d(inputs, filters = 64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_regularizer=regularizer)
      conv1p = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, padding='SAME')
      print (conv1p.shape)
      conv2 = tf.layers.conv1d(conv1p, filters = 128, kernel_size=3,  activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_regularizer=regularizer)
      conv2p = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding='SAME')
      print (conv2p.shape)

      flat = tf.layers.flatten(conv2p)
      print (flat.shape)

      lay1_D = tf.layers.dense(flat, 512, activation=None)
      lay2_D = tf.layers.dense(lay1_D, 256, activation=None)
      lay3_D = tf.layers.dense(lay2_D, 128, activation=None)
      print (lay3_D.shape)
      out = tf.layers.dense(lay3_D, 1, activation=None)

      return out

  def generator(self, inputs, name='G_model'):
    # inputs = tf.reshape(inputs, [self.bs, 128, 512, 1])
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    with tf.variable_scope(name) as scope:

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
              activation=tf.nn.relu, kernel_regularizer=regularizer)   
      conv3 = tf.layers.dropout(conv3, self.keep_prob_var)
      
      conv3p = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv3p.shape)

      conv4 = tf.layers.conv2d(conv3p, filters=64, kernel_size=(3,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              activation=tf.nn.relu, kernel_regularizer=regularizer)
      conv4 = tf.layers.dropout(conv4, self.keep_prob_var)

      conv4p = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      print (conv4p.shape)

      # conv5 = tf.layers.conv2d(conv4p, filters=256, kernel_size=(3,3), kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
      #         activation=tf.nn.relu, kernel_regularizer=regularizer)
      # conv5 = tf.layers.dropout(conv5, self.keep_prob_var)
      # conv5p = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
      # print (conv5p.shape)

      flat = tf.layers.flatten(conv4p)
      print (flat.shape)
      lay1 = tf.layers.dense(flat, 4096, activation=None)
      # lay1 = tf.layers.dropout(lay1, 0.2)
      lay2 = tf.layers.dense(lay1, 2048, activation=None)
      # lay2 = tf.layers.dropout(lay2, 0.2)    
      lay3 = tf.layers.dense(lay2, 1024, activation=None)
      # lay1 = tf.layers.dropout(lay1, 0.2)  
      output = tf.layers.dense(lay3, 50, activation=None)

      return output

  def train(self, sess):
    # tf.initialize_all_variables().run()
    print ('initializing...opt')
    d_opt = self.d_opt
    g_opt = self.g_opt

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


    d_losses = []
    d_fake_losses = []
    d_real_losses = []
    g_losses = []
    g_adv_losses = []
    g_l1_losses = []
    v_l1_losses = []

    try:
      while not coord.should_stop():
        start = timeit.default_timer()       

        for i in range(2):
          _d_opt, d_fake_loss, d_real_loss = self.sess.run([d_opt, self.d_fake_losses[0], self.d_real_losses[0]], feed_dict={self.is_valid:False})
        _g_opt, g_adv_loss, g_l1_loss = self.sess.run([g_opt, self.g_adv_losses[0], self.g_l1_losses[0]], feed_dict={self.is_valid:False})
        v_l1_loss = self.sess.run(self.g_l1_losses[0], feed_dict={self.is_valid:True})
        
        end = timeit.default_timer()
        batch_timings.append(end - start)

        d_fake_losses.append(d_fake_loss)
        d_real_losses.append(d_real_loss)
        g_adv_losses.append(g_adv_loss)
        g_l1_losses.append(g_l1_loss)
        v_l1_losses.append(v_l1_loss)

        print('{}/{} (epoch {}), '
                      'd_rl_loss = {:.5f}, '
                      'd_fk_loss = {:.5f}, '
                      'g_adv_loss = {:.5f}, '
                      'g_l1_loss = {:.5f}, '
                      'valid_l1_loss = {:.5f}, '                      
                      ' time/batch = {:.5f}, '
                      'mtime/batch = {:.5f}'.format(counter,
                                                    self.epoch * num_batches,
                                                    current_epoch,
                                                    d_real_loss,
                                                    d_fake_loss,
                                                    g_adv_loss,
                                                    g_l1_loss,
                                                    v_l1_loss,
                                                    end - start,
                                                    np.mean(batch_timings)))


        if counter % 10 == 0:
          pred, y = self.sess.run([self.fake_input, self.rat], feed_dict={self.is_valid:False})
          print ('pred',pred)
          print ('y',y)

        batch_idx += 1
        counter += 1
        if (counter) % 2000 == 0 and (counter) > 0:
          self.saver.save(self.sess, "./only_spec_gan_"+str(counter)+"_th")
        if batch_idx >= num_batches:
          current_epoch += 1
          #reset batch idx
          batch_idx = 0
        if current_epoch >= self.epoch:
          print (str(self.epoch),': epoch limit')
          print ('saving last model at iteration',str(counter))
          self.saver.save(self.sess, "./only_spec_gan_final")
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