#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import argparse
import time
from keras import callbacks as cbks
from keras.datasets import mnist
import logging
import tensorflow as tf
import numpy as np
from server import client_generator

np.random.seed(123)
ims = [210, 160, 3]


def train_model(name, ftrain, generator, samples_per_epoch, nb_epoch,
                verbose=1, callbacks=[],
                ftest=None, test_data=None, validation_data=None, nb_val_samples=None,
                saver=None, gif=None):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    if gif:
      plt.subplot(121)
      IM = plt.imshow(np.random.uniform(0, 256, ims).astype('uint8'), interpolation="none")
      plt.subplot(122)
      IM2 = plt.imshow(np.random.uniform(0, 256, ims).astype('uint8'), interpolation="none")
      plt.draw()
      plt.pause(.001)

    self = {}
    epoch = 0
    counter = 0
    out_labels = ['loss', 'nll', 'time']  # self.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]
    train_costs = np.zeros((nb_epoch, 4))

    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + callbacks + [history]
    if verbose:
      callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)

    callbacks.set_params({
        'nb_epoch': nb_epoch,
        'nb_sample': samples_per_epoch,
        'verbose': verbose,
        'metrics': callback_metrics,
        'epochs': nb_epoch
    })
    callbacks.on_train_begin()

    while epoch < nb_epoch:
      callbacks.on_epoch_begin(epoch)
      samples_seen = 0
      batch_index = 0
      while samples_seen < samples_per_epoch:
        x, y = next(generator)
        x = x[:, :-1]
        y = x[:, 1:]
        # build batch logs
        batch_logs = {}
        if type(x) is list:
          batch_size = len(x[0])
        elif type(x) is dict:
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        t1 = time.time()
        samples, losses, nll_loss = ftrain(x, y, counter)
        train_costs[epoch, 0] += losses
        train_costs[epoch, 1] += nll_loss
        outs = (losses, nll_loss) + (time.time() - t1, )
        counter += 1

        if (counter % 100 == 0) and gif:
          for v, u in zip(samples[0], y[0]):
            IM.set_data(((v.reshape(ims)+1)*127.5).astype('uint8'))
            IM2.set_data(((u.reshape(ims)+1)*127.5).astype('uint8'))
            plt.draw()
            plt.pause(.01)

        for l, o in zip(out_labels, outs):
          batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        # construct epoch logs
        epoch_logs = {}
        batch_index += 1
        samples_seen += batch_size
    
      train_costs[epoch, :] = train_costs[epoch, :] / samples_per_epoch

      if saver is not None:
        saver(epoch)

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1

    # save_gif(samples, 
    out_costs = 0.
    if ftest is not None:
      val_seen = 0
      val_cost = 0
      test_cost = 0
      val_nll = 0
      test_nll = 0
      while val_seen < nb_val_samples:
        _, valc, v_nll_loss = ftest(*next(validation_data))
        _, talc, t_nll_loss = ftest(*next(test_data))
        val_cost += valc
        test_cost += talc
        val_seen += 1
      val_cost /= val_seen
      test_cost /= val_seen
      val_nll /= v_nll_loss
      test_nll /= t_nll_loss
      out_costs = val_cost, test_cost, val_nll, test_nll
      print "Val: ", val_cost, "Test: ", test_cost

    # _stop.set()
    callbacks.on_train_end() 
    np.save("./outputs/out_costs_"+name, [train_costs, out_costs])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generative model trainer')
  parser.add_argument('model', type=str, default="bn_model", help='Model definitnion file')
  parser.add_argument('--name', type=str, default="autoencoder", help='Name of the model.')
  # parser.add_argument('--time', type=int, default=1, help='How many temporal frames in a single input.')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--batch', type=int, default=32, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--time', type=int, default=10, help='Length of the temporal series')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--loadweights', dest='loadweights', action='store_true', help='Start from checkpoint.')
  parser.add_argument('--gif', dest='gif', action='store_true', help='Visualize during training.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  parser.set_defaults(gif=False)
  args = parser.parse_args()

  MODEL_NAME = args.model
  logging.info("Importing get_model from {}".format(args.model))
  exec("from models."+MODEL_NAME+" import get_model")
  # try to import `cleanup` from model file


  model_code = open('models/'+MODEL_NAME+'.py').read()

  if not os.path.exists("./outputs/results_"+args.name):
    os.makedirs("./outputs/results_"+args.name)
  if not os.path.exists("./outputs/samples_"+args.name):
    samples_path = "./outputs/samples_"+args.name
    os.makedirs(samples_path)
  if not os.path.exists("./outputs/logs_"+args.name):
    os.system("rm -rf ./outputs/logs_"+args.name)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  with tf.Session() as sess:
    ftrain, ftest, loader, saver, extras = get_model(sess=sess, name=args.name, batch_size=args.batch, time_len=args.time)

    # start from checkpoint
    if args.loadweights:
      loader()

    train_gen = client_generator(hwm=20, host=args.host, port=args.port)

    train_model(args.name, ftrain,
                train_gen,
                samples_per_epoch=args.epochsize,
                ftest=None, validation_data=None, test_data=None, nb_val_samples=100,
                nb_epoch=args.epoch, verbose=1, saver=saver, gif=args.gif
                )
