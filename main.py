'''
Purpose: A digit classifier made in Python using a neural network.

Owen Colley
9/3/24
'''

import numpy

i_nodes = 2
o_nodes = 2
h_nodes = 3
batch_size = 8

b1 = numpy.zeros((1, h_nodes))
b2 = numpy.zeros((1, o_nodes))

i_data = numpy.random.randn(batch_size, i_nodes)
o_data = numpy.random.randn(batch_size, o_nodes)

w1 = numpy.random.randn(i_nodes, h_nodes)
w2 = numpy.random.randn(h_nodes, o_nodes)

loss_array = numpy.array([[]])
indices = numpy.array([[]])

for i in range(1000):
  h_values = i_data.dot(w1) + b1
  h_relu = numpy.maximum(h_values, 0)
  o_data_predictions = h_relu.dot(w2) + b2
  
  loss = numpy.square(o_data_predictions - o_data).sum()
  loss_array = numpy.append(loss_array, loss)
  indices = numpy.append(indices, i)
  
  grad_pred = 2 * (o_data_predictions - o_data)
  grad_w2 = h_relu.T.dot(grad_pred)
  grad_b2 = grad_pred.sum(axis=0, keepdims=True)
  grad_h_relu = grad_pred.dot(w2.T)
  grad_h_values = grad_h_relu.copy()
  grad_h_calues[h_values < 0] = 0
  grad_w1 = i_data.T.dot(grad_h_values)
  grad_b1 = grad_h_values.sum(axis=0, keepdims=True)
  
  w1 -= grad_w1 * 1e-4
  b1 -= grad_b1 * 1e-4
  w2 -= grad_w2 * 1e-4
  b2 -= grad_b2 * 1e-4

plt.plot(indices, loss_array)
plt.legend(["Loss over iterations"])
plt.show()

print(o_data)
print(o_data_predictions)
