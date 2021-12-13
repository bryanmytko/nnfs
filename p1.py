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











































