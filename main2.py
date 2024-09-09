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

i_data = numpy.random.randn(batch_size, i_nodes)
o_data = numpy.random.randn(batch_size, o_nodes)

w1 = numpy.random.randn(i_nodes, h_nodes)
w2 = numpy.random.randn(h_nodes, o_nodes)

loss_array = numpy.array([[]])
indices = numpy.array([[]])

for i in range(1000):
  h_values = i_data.dot(w1)
  h_relu = numpy.maximum(h_values, 0)
  o_data_predictions = h_relu.dot(w2)
  
  loss = numpy.square(o_data_predictions - o_data).sum()
  loss_array = numpy.append(loss_array, loss)
  indices = numpy.append(indices, i)
  
  grad_pred = 2 * (o_data_predictions - o_data)
  grad_w2 = h_relu.T.dot(grad_pred)
  grad_h_relu = grad_pred.dot(w2.T)
  grad_h_values = grad_h_relu.copy()
  grad_h_calues[h_values < 0] = 0
  grad_w1 = i_data.T.dot(grad_h_values)
  
  w1 -= grad_w1 * 1e-4
  w2 -= grad_w2 * 1e-4

plt.plot(indices, loss_array)
plt.legend(["Loss over iterations"])
plt.show()

print(o_data)
print(o_data_predictions)
