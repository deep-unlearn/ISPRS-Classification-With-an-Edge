import matplotlib.pyplot as plt
import h5py
import caffe
import numpy as np

"""
Method for visualizing all channels in a 2D grid - Can be used in real-time during training for visualizing states of particular blobs.
"""

def vis(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max() + 1e-8

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.show(plt.imshow(data, cmap='gist_ncar'))
