import os
import numpy as np
from scipy.misc import imread, imresize
from random import sample, seed
from glob import glob
from matplotlib import pyplot as plt
from vgg19 import VGG19
from tqdm import tqdm
seed(1337)
np.random.seed(1337)
avg_max = 7321.79384766

def get_data(filename='./train.split.1', batch_size=16, time_len=16, image_size=(224, 224, 3)):
    time_len += 1
    vgg = VGG19()
    folders = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            name, label = l.split(' ')
            label = int(label)
            folders.append([name, label])

    while True:
        batch_folders = sample(folders, batch_size)
        X = np.zeros((batch_size, time_len) + image_size)
        Y = np.zeros((batch_size, time_len, 1))
        for i, bf in enumerate(batch_folders):
            imfiles = glob(os.path.join(bf[0], '*'))
            if len(imfiles) < time_len:
                imfiles.extend(imfiles)
            start_idx = np.random.randint(0, max(len(imfiles)-time_len, 1), 1)[0]
            imfiles = imfiles[start_idx:start_idx+time_len]
            crop_range = np.random.randint(0, 256-image_size[0], 1)[0]
            for j, imf in enumerate(imfiles):
                I = imread(imf)
                r, c, _ = I.shape
                if r > c:
                    I = imresize(I, [int(256*r/c), 256])
                else:
                    I = imresize(I, [256, int(256*c/r)])
                I = I[crop_range:crop_range+image_size[0], crop_range:crop_range+image_size[1]]
                X[i, j, :, :, :] = I
                Y[i, j, :] = bf[1]
        out = vgg.predict(X.reshape((-1,)+image_size), batch_size=32)[0]
        out = out.reshape((batch_size, time_len)+out.shape[1:])
        out = out/avg_max - .5
        yield out[:, :-1], Y


if __name__ == '__main__':
    dg = get_data()
    m = 0
    for i in tqdm(range(20)):
        X, Y = next(dg)
        print X.shape, Y.shape, X.mean(), abs(X).min(), X.max(), X.std()
        m += abs(X).max()
    print m/10.

