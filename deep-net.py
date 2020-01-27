# using compat for running tensorflow 2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# importing input_data some kind of input reader from tensorflow for specifically mnist?
from tensorflow.examples.tutorials.mnist import input_data

# reading data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# defining number of neurons in hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of neurons in output layer
n_classes = 10
batch_size = 100


# defining the x and y axis, flattening the 28x28 into [None, 784] tensor
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


# neural network; creating the model
def neural_network_model(data):
    # defining weights and biases on hidden layers
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input data * weight) + bias
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # threshold function

    # (input data * weight) + bias
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) # threshold function

    # (input data * weight) + bias
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) # threshold function

    # input data + bias
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

# train
def train_neural_network(x):
    # build the model
    prediction = neural_network_model(x)
    # using cost function softmax_cross_entropy_with_logits
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # using optimizer to backpropagate and optimize weights on the layers set to minimize cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # feed forward + backpropagation  = epoch
    hm_epochs = 10

    # execute
    with tf.Session() as sess:
        # initialize the variables (not sure what this means)
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict= {x: epoch_x, y: epoch_y })
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({ x: mnist.test.images, y: mnist.test.labels }))

train_neural_network(x)
