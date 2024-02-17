# Implements random mini batch

import math
import h5py
import numpy as np
import matplotlib.pyplot as plt


%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def random_mini_batch(X, Y, mini_batch_size=32):
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation].reshape((1, m))

    batch_num = math.floor(m/mini_batch_size)

    for i in range(0, batch_num):
        mini_batch_X = X_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]
        mini_batch_y = Y_shuffled[:, i*mini_batch_size : (i+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    


