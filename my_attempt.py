'''
Purpose: A digit classifier made in Python using a neural network.

Owen Colley
9/3/24

REMEMBER TO TRY TO MAKE BITMASK NEURAL NETWORK
'''

import numpy

# define the size of the neural network
input_nodes = 2
hidden_nodes = 3
output_nodes = 2
batch_size = 8
batches = 1000

# defines input data and output data for testing and training model
input_data = numpy.random.randn(batch_size, input_nodes)
output_data = numpy.random.randn(batch_size, output_nodes)

# set weight arrays to random number (setting to 0 could cause a dead system)
weights1 = numpy.random.randn(input_nodes, hidden_nodes)
weights2 = numpy.random.randn(h_nodes, output_nodes)

# set bias arrays to 0
biases1 = numpy.zeros((1, hidden_nodes))
biases2 = numpy.zeros((1, output_nodes))

# set empty loss array and indices to fill when training data
loss_array = numpy.array([[]])
indices = numpy.array([[]])

# loop for training iterations
for i in range(batches):

  # loop to account for every forward pass of data, allowing my complex pattern recognition
  for a in range(num_hidden_layers):
    # define the hidden values using input * weight + bias (y = mx + b)
    hidden_values = input_data.dot(weights1) + biases1
  
    # add non linearity to the line by seting everything below 0 to 0. Bias stops this from causing problems
    hidden_relu_values = numpy.maximum(0, hidden_values)

  # add loss found using (guess - actual)^2 which amplifies errors, adds it to array along with its index
  loss = numpy.square(output_data_predictions - output_data).sum()
  loss_array = numpy.append(loss_array, loss)
  indices = numpy.append(indices, i)

  # uses derivative of loss to find the gradient predicion
  gradient_pred = 2 * (output_data_predictions - output_data)

  # tr
  gradient_w2 = hidden_relu_values.T.dot(grad_pred)
  gradient_b2 = grad_pred.sum(axis=0, keepdims=True)
  grad_h_relu = grad_pred.dot(w2.T)
  grad_h_values = grad_h_relu.copy()
  grad_h_values[h_values < 0] = 0
  grad_w1 = i_data.T.dot(grad_h_values)
  grad_b1 = grad_h_values.sum(axis=0, keepdims=True)

  # shifts weights and biases slightly closer to solution using slope to eventually find minimum
  weights -= grad_w1 * 1e-4
  biases -= grad_b1 * 1e-4
  w2 -= grad_w2 * 1e-4
  b2 -= grad_b2 * 1e-4

print("Output Data: ", output_data)
print("Output Predictions: ", output_data_predictions)
