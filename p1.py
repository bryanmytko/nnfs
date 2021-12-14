import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

# Define a hidden layer

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # correct shape, no transpose needed!
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# inputs (3,4) 3 neurons, 4 input size
# weights (4, x)
#
# inputs (4, 
#
# O ----  x --- x
# O       x     x
# O       x
#         x

layer1 = Layer_Dense(4,5) # 4 inputs, can be any number of neurons
# output looks like: 
# [[x,x,x,x,x]
# [x,x,x,x,x]
# [x,x,x,x,x]]
# because X was (3, 4) and we had 5 neurons for this layer

layer2 = Layer_Dense(5,2) # 5 inputs, any number of neurons
# output looks like: 
# [[x,x]
# [x,x]
# [x,x]]
# because outputX (3,5) and we had 2 neurons for this layer

layer1.forward(X)
print('layer one output: ', layer1.output)

layer2.forward(layer1.output)
print('layer two output: ', layer2.output)

# Using Activation Functions
# More: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # correct shape, no transpose needed!
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Rectified Linear Unit function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)

print(layer1.output) # We see a bunch of values < 0, haven't ran through ReLU yet

activation1.forward(layer1.output)
print(activation1.output) #!!!!
