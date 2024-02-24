# Implements random mini batch

import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets



# %matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def update_parameters_with_gd(parameters, grads, learning_rate=0.01):
    L = len(parameters)//2
    for l in range(1, L):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate*grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate*grads['db' + str(l)]
    return parameters

def random_mini_batch(X, Y, mini_batch_size=32):
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
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
    W1 = parameters['W1']
    b1 = parameters['b1']    
    W2 = parameters['W2']    
    b2 = parameters['b2']    
    W3 = parameters['W3']    
    b3 = parameters['b3']

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3   
    a3 = sigmoid(z3)

    cache = (W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, a3)
    return a3, cache


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, a3) = cache

    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)
    
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients

def model(X, Y, layer_dims, optimizer, learning_rate=0.01, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000):
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
                parameters = update_parameters_with_adam(parameters, gradients, learning_rate, v, s, t, beta1, beta2, epsilon)

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

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int64)
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def predict_dec(parameters, X):
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


train_X, train_Y = load_dataset()
layer_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layer_dims, optimizer='gradient_descent')
predictions = predict(train_X, train_Y, parameters)

plt.title("model with gradient descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)