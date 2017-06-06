"""
Fully connected model, double digit, sigmoid
"""
import os
import time
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Reshape, Conv2D, Lambda, Input, Activation, Layer, Deconv2D
from keras.objectives import binary_crossentropy as bce
from utils import batch_conv2d, load, save, Background, Delta2, uniform, wta
K.set_image_dim_ordering('tf')
learning_rate = .001
beta1 = .9
obj_size = 8
latent_dim = 100
epsilon_std = 0.01
n_h = 512
K.set_image_dim_ordering('tf')


def get_model(sess, name, time_len=10, batch_size=16, image_size=(14, 14, 512)):
    checkpoint_dir = './outputs/results_' + name
    with tf.variable_scope(name):
	# Inputs
        X = Input(batch_shape=(batch_size, time_len)+image_size)
        Y = Input(batch_shape=(batch_size, time_len)+image_size)

	# Conv encoder
	enc = Sequential()
	enc.add(Conv2D(512, kernel_size=(5, 5), padding='same', input_shape=image_size))
	enc.add(Activation('relu'))
	enc.add(Conv2D(512, kernel_size=(5, 5), padding='same'))
	enc.add(Activation('relu'))

	# Conv decoder
	dec = Sequential()
        dec.add(Conv2D(512, kernel_size=(7, 7), padding='same', input_shape=image_size[:2]+(512,)))
  	
	# RNNs
	ix = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_0')
	ih = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_1')
	fx = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_2')
	fh = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_3')
	ox = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_4')
	oh = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_5')
	cx = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_6')
	ch = Conv2D(512, kernel_size=(5, 5), padding='same', name='rnn_7')

        h_0 = tf.zeros(shape=(batch_size,)+image_size[:2]+(512,), dtype=tf.float32)
        c_0 = tf.zeros(shape=(batch_size,)+image_size[:2]+(512,), dtype=tf.float32)
	
        def step(x, states):
            # LSTM
            h, c = states[0], states[1]
            i = tf.nn.sigmoid(ix(x) + ih(h))
            f = tf.nn.sigmoid(fx(x) + fh(h))
            o = tf.nn.sigmoid(ox(x) + oh(h))
            c_t = f * c + i * tf.nn.tanh(cx(x) + ch(h))
            h_t = o * tf.nn.tanh(c_t)

            return h_t, [h_t, c_t]


	# warmup
        Xr = tf.reshape(X, (-1,)+image_size)
        z_r = enc(Xr)
        z_r = tf.reshape(z_r, (-1, time_len, 14, 14, 512))
        last_, Out, states = K.rnn(step, z_r, initial_states=[h_0, c_0], constants=[], input_length=time_len, unroll=True)
        Pre = tf.reshape(Out, (-1,)+image_size[:2]+(512,))
        Out = wta(Pre)
        X_hat = dec(Out)

        mvars = [v for v in tf.global_variables() if "rnn_" in v.name] + enc.trainable_weights + dec.trainable_weights
        print mvars

        Yr = tf.reshape(Y, (-1,) + image_size)
        cost = tf.reduce_mean(tf.square(Yr-X_hat))
        optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(cost, var_list=mvars)
        tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter("./outputs/logs_{}".format(name), K.get_session().graph)
    saver = tf.train.Saver()
    sum_loss = tf.summary.scalar("loss", cost)
    sum_gen = tf.summary.image("X_hat", tf.reshape(X_hat, (-1,)+image_size)[:, :, :, :3])
    sum_weights = tf.summary.image("W", tf.transpose(dec.layers[0].kernel, (2, 0, 1, 3))[:, :, :, :3])
    print(dec.layers[0].kernel)
    sums = tf.summary.merge_all()

    def ftrain(x, y, counter, sess=sess):
        outs, loss, wsums, _ = sess.run([X_hat, cost, sums, optim], feed_dict={
            X:x, Y:y})
        writer.add_summary(wsums, counter)
        outs = outs.reshape((batch_size, time_len) + image_size)
        return outs, loss, 0.

    def ftest(x, y, sess=sess):
        outs, loss = sess.run([X_hat, cost], feed_dict={X:x, Y: y})
        return outs, loss, 0.

    def fextract(x, sess=sess):
        outs, = sess.run([Pre], feed_dict={X:x})
        return outs

    def f_load():
        load(sess, saver, checkpoint_dir, name)

    def f_save(step):
        save(sess, saver, checkpoint_dir, 0, name)

    return ftrain, ftest, f_load, f_save, [fextract]
