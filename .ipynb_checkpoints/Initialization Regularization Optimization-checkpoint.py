# -*- coding: utf-8 -*-
"""Initialization, regularization and optimization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ya8QxrmRzVsuX2ctFKlZYSGoDezUm_i5

**End to end neural net practice**

###Objectives:
1.   Difference in results with different initialization values i.e random, zeroes and He initialization.
2.   Add regularization to reduce overfitting
3.   Optimize gradient descent using momentum, RMS prop and ADAM
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# %matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['figure.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X_train, Y_train, X_test, Y_test = load_data()

"""###Functions"""

def load_data():
  dd = load_diabetes()
  X_OG = dd.data
  Y_OG = dd.target
  X_train, X_test, Y_train, Y_test = train_test_split(X_OG, Y_OG, test_size=0.2, random_state=42)
  return X_train, Y_train, X_test, Y_test

def initialize_parameters(layer_dims, type="random"):
  L = len(layer_dims)
  for l in range(1, L):
    if type == 'random':
      parameters['W' + str(l)] = np.random.randn((layer_dims[l], layer_dims[l-1])) * 10
    elif type == 'zeroes':
      parameters['W' + str(l)] = np.zeroes((layer_dims[l], layer_dims[l-1]))
    elif type == "he":
      parameters['W' + str(l)] = np.random.randn((layer_dims[l], layer_dims[l-1])) * np.sqrt(2/layer_dims[l-1])
    parameters['b' + str(l)] = np.zeroes((layer_dims[l], 1))
  return parameters

def sigmoid(Z):
  A = 1/(1 + np.exp(-Z))
  cache = A
  return A, cache

def relu(Z):
  A = np.maximum(0, Z)
  cache = A
  return A, cache

def relu_backwards(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backwards(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def calculate_cost(AL, Y):
  m = AL.shape
  J = (1/m)* (-np.dot(Y, np.log(AL).T) - np.dot((1-Y), np.log(1 - AL).T))
  return np.squeeze(J)

def update_parameters(parameters, grads, learning_rate=0.01):
  L = parameters//2
  for l in range(1, L):
    parameters['W' +  str(l)] = learning_rate * grads['dW' + str(l)]
    parameters['b' +  str(l)] = learning_rate * grads['db' + str(l)]
  return parameters

"""###Forward pass"""

def forward_pass(X, parameters):
  L = parameters//2
  caches = []
  A = X
  for l in range(1, L):
    A_prev = A
    Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
    A, cache = relu(Z)
    cache.append(Z)

  Z = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
  AL, cache = sigmoid(Z)
  cache.append(Z)

  caches.append(cache)

  return AL, caches

"""###Backward propagation"""

def linear_backwards(dZ, cache):
  A_prev, W, b = cache
  m = A_prev.shape[1]

  dW = 1./m * np.dot(dZ,A_prev.T)
  db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
  dA_prev = np.dot(W.T,dZ)

  return dA_prev, dW, db


def linear_backward_activation_function(dA, cache, activation_function):
  linear_cache, activation_cache = cache
  if activation_function == 'sigmoid':
    dZ = sigmoid_backwards(dA, activation_cache)
    dA_prev, dW, db = linear_backwards(dZ, linear_cache)
  elif activation_function == 'relu':
    dZ = relu_backwards(dA, activation_cache)
    dA_prev, dW, db = linear_backwards(dZ, linear_cache)

  return dA_prev, dW, db


def backward_prop(AL, Y, cache):
  grads = []
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)
  L = len(cache)

  cur_cache = cache[L-1]
  dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

  grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward_activation_function(dAL, cur_cache, activation_function="sigmoid")

  for l in reversed(range(L-1)):
    cur_cache = caches[l]
    grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_backward_activation_function('dA' + str(l + 2), cur_cache, activation_function="relu")

  return grads

"""###Model"""

def nn_model(X, Y, layer_dims, learning_rate=1.2, iterations=1000, initialization="random"):
  costs = []
  parameters = initialize_parameters(layer_dims)
  for iter in range(iterations):
    AL, cache = forward_pass(X, parameters)
    cost = calculate_cost(AL, Y)
    grads = backward_prop(AL, Y, cache)
    parameters = update_parameters(parameters, grads, learning_rate)

    if cost%100 == 0:
      costs.append(cost)

  plt.plot(np.squeeze(cost))
  plt.ylabel('Cost')
  plt.xlabel('Iterations')
  plt.title('Learning Rate')
  plt.show()

  return parameters

"""###Trigger"""
