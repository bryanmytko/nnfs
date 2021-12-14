# One neuron

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0] * weights[0] + \
    inputs[1] * weights[1] + \
    inputs[2] * weights[2] + \
    inputs[3] * weights[3] + \
    bias

print('Single neuron:', output)

# Three  neurons
#
# this time we need 3 sets of weights
# also three bias

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [
        inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] +inputs[3] * weights1[3] + bias1,
        inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] +inputs[3] * weights2[3] + bias2,
        inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] +inputs[3] * weights3[3] + bias3
        ]

print('Three neurons:', output)

# Refactor

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print('Refactor:', layer_outputs)

# Shapes

# Array:
#     l = [1, 5, 6, 2]
# Shape:
#     (4,)
# Type:
#     1-d array, or a Vector
#
# ----------------------------
#
# Array:
#     l = [[1, 5, 6, 2],
#         [1, 5, 6, 2]]
# Shape:
#     (2,4)
# Type:
#     2-d array, or a Matrix (rectangular array)
#
# ----------------------------
#
# Array:
#     l = [
#             [
#                 [1, 5, 6, 2],
#                 [1, 5, 6, 2]
#             ],
#             [
#                 [1, 5, 6, 2],
#                 [1, 5, 6, 2]
#             ],
#             [
#                 [1, 5, 6, 2],
#                 [1, 5, 6, 2]
#             ]
#         ]
# Shape:
#     (3, 2, 4)
# Type:
#     3-d array
#
# ----------------------------
#
# Tensor is an objec that _can_ be represented as an array

import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
biases = 2.0

output = np.dot(weights, inputs) + bias # weights must come first
print('Dot product of two vectors: ', output)

# Dot product: (gives scaler)
#
#     a = [1,2,3]
#     b = [2,3,4]
#
#     dot_product = a[0]*b[0] + ...
#     or
#     *a* dot *b* = 1*2 + 2*3 + 3*4 = 20

# Dot product of a layer:
# Note this is bunch of dot products of each vector, which is different than just "dot product"
# This type of matrix multiplication is _not_ commutitive (1xn x nx1)
# Matrix multiplication is actually just a composition of linear transformations

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print('Dot product of the layer:', output)

# Batches
#
# we can calculate these things in parallel
# doing many things in parallel; this is why we run these on the gpu

inputs = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

# This won't work because the shape of these inputs and weights are the same so we can't dot product
# (It only worked previously because the inputs were a vector not a matrix)
# Solution is the transpose the weights array

biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases
print('Dot product of the first batch:', output)

# Adding another layer. 

weights2 = [[0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print('Layer 2 outputs, manually: ', layer2_outputs)

# This quickly becomes unruly, manually adding each layer. Let's use objects!

# input feature set is denoted by capital X
# in this example our input data is not normalized
X = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0) # keep random values the same

# define the hidden layers

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

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

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
print(activation1.output)















