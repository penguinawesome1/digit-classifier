'''
Purpose: A digit classifier made in Python using a neural network.

Owen Colley
9/3/24
'''

import numpy

# parameters for the network

i_nodes = 2
o_nodes = 2
h_nodes = 3
batch_size = 8

i_data = numpy.random.randn(batch_size, i_nodes)
o_data = numpy.random.randn(batch_size, o_nodes)

w1 = numpy.random.randn(i_nodes, h_nodes)
w2 = numpy.random.randn(h_nodes, o_nodes)

h_values = i_data.dot(w1)

# Rectified Linear Unit (ReLU)
# removes all negative values, replacing them with 0

h_relu = numpy.maximum(h_values, 0)
o_data_predictions = h_relu.dot(w2)

loss = numpy.square(o_data_predictions - o_data).sum()
