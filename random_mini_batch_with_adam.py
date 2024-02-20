# Implements random mini batch

import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets



%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate=0.01):
    L = len(parameters)//2
    for l in range(1, L):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate*grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate*grads['db' + str(l)]
    return parameters

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

def load_dataset():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=0.2)
    print(train_X.shape, train_X)
    print(train_X[:, 0])
    plt.scatter(train_X[:, 0], train_X[:,1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
        
def calculate_cost(a3, Y):
    J = (np.multiply(Y, -np.log(a3))) + np.multiply((1 - Y), -np.log(1 - a3))
    cost_total = np.sum(J)
    return cost_total

def forward_propagation(X, parameters):


def backward_propagation(X, Y, caches):


def nn_model(X, Y, layer_dims, optimizer, learning_rate=0.01, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000):
    L = len(layer_dims)
    costs = []
    t = 0
    m = X.shape[1]
    seed = 5
    parameters = initialize_parameters(layer_dims)

    if optimizer == 'gradient_descent':
        pass
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)
    
    for i in range(num_epochs):
        seed += 1
        minibatches = random_mini_batch(X, Y, mini_batch_size)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)

            cost_total += calculate_cost(a3, minibatch_Y)

            gradients = backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == 'gradient_descent':
                parameters = update_parameters_with_gd(parameters, gradients, learning_rate)
            elif optimizer == 'momentum':
                parameters = update_parameters_with_momentum(parameters, gradients, v, learning_rate, beta)
            elif optimizer == 'adam':
                t += 1
                parameters = update_parameters_with_gd(parameters, gradients, learning_rate, v, s, t, beta1, beta2, epsilon)

        cost_avg = cost_total / m
        if i%1000 == 0:
            print("Cost after epoch %i: %f" %(i, cost_avg))

        if i%100 == 0:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch (per 100)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters




train_X, train_Y = load_dataset()
