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

    # Handle for the remaining examples
    if (m%mini_batch_size != 0):
        mini_batch_X = X_shuffled[:, int(m/mini_batch_size)*mini_batch_size : ]
        mini_batch_Y = Y_shuffled[:, int(m/mini_batch_size)*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches    


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v['dW' + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1]))
        v['db' + str(l+1)] = np.zeros((parameters['b' + str(l+1)].shape[0], parameters['b' + str(l+1)].shape[1]))

    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters)//2
    for l in range(1, L):
        v['dW' + str(l)] = (beta * v['dW' + str(l)]) + ((1- beta) * grads['dW' + str(l)])
        v['db' + str(l)] = (beta * v['db' + str(l)]) + ((1 - beta) * grads['db' + str(l)])

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v['db' + str(l)]

    return parameters, v

def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}
    for l in range(1, L):
        v['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        v['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))

        s['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        s['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))

    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, beta1=0.9, beta2=0.999, learning_rate=0.01, epsilon=1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}
    for l in range(1, L):
        v['dW' + str(l)] = beta1*v['dW' + str(l)] + ((1 - beta1)*v['dW' + str(l)])
        v['db' + str(l)] = beta1*v['db' + str(l)] + ((1 - beta1)*v['db' + str(l)])

        v_corrected['dW' + str(l)] = np.divide(v['dW' + str(l)], (1 - np.exp(beta1, t)))
        v_corrected['db' + str(l)] = v['db' + str(l)]/(1 - beta1**t)

        s['dW' + str(l)] = beta2*s['dW' + str(l)] + ((1 - beta2)*np.square(s['dW' + str(l)]))
        s['db' + str(l)] = beta2*s['db' + str(l)] + ((1 - beta2)*np.square(s['db' + str(l)]))

        s_corrected['dW' + str(l)] = s['dW' + str(l)]/(1 - beta2**t)
        s_corrected['db' + str(l)] = s['db' + str(l)]/(1 - beta2**t)

        parameters['dW' + str(l)] = parameters['dW' + str(l)] - learning_rate*(v_corrected['dW' + str(l)]/(np.sqrt(s_corrected['dW' + str(l)]) + epsilon))
        parameters['db' + str(l)] = parameters['db' + str(l)] - learning_rate*(v_corrected['db' + str(l)]/(np.sqrt(s_corrected['db' + str(l)]) + epsilon))

    return parameters, v, s


