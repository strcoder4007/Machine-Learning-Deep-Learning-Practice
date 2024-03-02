import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

# Create an input tensor using tf.keras.layers.Input
input_tensor = Input(shape=(32, 32, 3))

# Create a dense layer using tf.keras.layers.Dense
dense_layer = Dense(32, activation='relu')(input_tensor)

# Print the values of the dense layer
tf.print(dense_layer)