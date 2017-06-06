import os
import tensorflow as tf
import numpy as np
from keras.engine.topology import Layer
from keras import backend as K
from scipy.linalg import toeplitz


def pack_outputs(lst, batch_size):
  new_outs = [tf.reshape(c, (batch_size, -1)) for c in lst]
  return K.concatenate(new_outs, -1)


def unpack_outputs(lst, latent_dim, image_size):
  out = lst[:, :, :image_size**2]
  m0, v0, m1, v1 = tf.split(2, 4, lst[:, :, image_size**2:])
  m0 = tf.reshape(m0, (-1, latent_dim))
  v0 = tf.reshape(v0, (-1, latent_dim))
  m1 = tf.reshape(m1, (-1, latent_dim))
  v1 = tf.reshape(v1, (-1, latent_dim))
  return out, m0, v0, m1, v1


def kl_cost(m0, v0, m1, v1):
  loss0 = - 0.5 * tf.reduce_mean(1 + v0 - K.square(m0) - K.exp(v0))
  loss1 = - 0.5 * tf.reduce_mean(1 + v1 - K.square(m1) - K.exp(v1))
  return loss0 + loss1


def batch_conv2d(input_, w, batch_size, W_shape=(28, 28, 1, 1)):
    print input_, w
    X = tf.split(0, batch_size, input_)
    W = tf.split(0, batch_size, w)
    outs = []
    for x, ww in zip(X, W):
        ww = tf.reshape(ww, W_shape) 
        conv = tf.nn.conv2d(x, ww, strides=[1, 1, 1, 1], padding='SAME')
        outs.append(conv)
    conv = tf.concat(0, outs)
    return conv


def wta(x):
    x = x * K.cast(K.equal(x, K.max(x, axis=(1, 2), keepdims=True)), "float32")
    return x


class Background(Layer):
    def __init__(self, canvas_size=(1, 64, 64, 1), **kwargs):
        self.canvas_size = canvas_size
        super(Background, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bg = K.zeros(self.canvas_size, name="background_{}".format(self.name))
        # bg = np.linspace(0, 1, self.canvas_size[1])
        # self.bg = K.variable(toeplitz(bg).reshape(self.canvas_size), name="backgroud")

        self.trainable_weights = [self.bg, ]

    def call(self, lst, mask=None):
        if not (isinstance(lst, list) or isinstance(lst, tuple)):
            lst = [lst, ]
        bg = K.reshape(self.bg, (1, )+self.canvas_size)
        for s in lst:
            # s += K.random_normal_variable(self.canvas_size, 0, .0001)
            # mask = K.cast(K.greater(s, 1e-4), "float32")  # non zero values
            # bg = s * mask + bg * (1-mask)
            # bg = s + bg * (1-s)
            bg = s
        return bg

    def get_output_shape_for(self, input_shape):
        try:
            return input_shape[0][:2] + self.canvas_size[1:]
        except:
            return input_shape[:2] + self.canvas_size[1:]


class Delta(Layer):
    def __init__(self, canvas_size=(64, 64), **kwargs):
        self.canvas_size = canvas_size
        super(Delta, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        N1, N2 = self.canvas_size
        a = x[:, 0:1]  # x coord
        b = x[:, 1:2]  # y coord
        s = 2.  # 2 * K.exp(x[:, 2:3])
        # gamma = x[:, 2:3]  # rescale
        grid_x = K.reshape(K.cast(tf.range(N1), tf.float32), [1, -1]) - N1/2. - .5
        grid_y = K.reshape(K.cast(tf.range(N2), tf.float32), [1, -1]) - N2/2. - .5
        px = K.exp(-K.square(grid_x - a) / s)
        px = K.reshape(px, (-1, N1, 1))
        py = K.exp(-K.square(grid_y - b) / s)
        py = K.reshape(py, (-1, 1, N2))
        P = K.batch_dot(px, py)
        P = P/K.maximum(K.sum(P, axis=(1, 2), keepdims=True), 1e-4)
        P = K.reshape(P, (-1, N1, N2, 1))
        # P = P * K.cast(K.equal(P, K.max(P, axis=(1, 2), keepdims=True)), "float32")
        return P

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + self.canvas_size + (1,)


class Delta2(Layer):
    def __init__(self, canvas_size=(64, 64), chan=1, **kwargs):
        self.canvas_size = canvas_size
	self.chan = chan
        super(Delta2, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        N1, N2 = self.canvas_size
        px = K.reshape(tf.nn.softmax(x[:, :self.canvas_size[0]]), (-1, self.canvas_size[0], 1))
        py = K.reshape(tf.nn.softmax(x[:, self.canvas_size[0]:]), (-1, 1, self.canvas_size[1]))
        P = K.batch_dot(px, py)
        P = K.reshape(P, (-1, N1, N2, 1))
        return P

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + self.canvas_size + (self.chan,)


def uniform(shape, scale=1., name=None):
    return K.random_uniform_variable(shape, 0.01, scale, name=name)


def save(sess, saver, checkpoint_dir, step, name):
  """Save tensorflow model checkpoint"""
  model_name = name
  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, saver, checkpoint_dir, name):
  """Load tensorflow model checkpoint"""
  print(" [*] Reading checkpoints: {}".format(checkpoint_dir))

  model_dir = name
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Checkpoints read: {}".format(ckpt_name))
    return True
  else:
    print(" [!] Failed reading.")
    return False
