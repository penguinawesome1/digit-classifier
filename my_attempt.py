'''
Purpose: A digit classifier made in Python using a neural network.

Owen Colley
9/3/24

REMEMBER TO TRY TO MAKE BITMASK NEURAL NETWORK
'''

import pandas
import numpy

'''df = pandas.read_csv('5year_stock.txt', header=0)

df.fillna(method='ffill', inplace=True)

df['Date'] = pandas.to_datetime(df['Date'], format='%m/%d/%Y')
df['Date'] = (df['Date'] - df['Date'].min()).dt.days

numerical_cols = ['Close/Last', 'Open', 'High', 'Low']
df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())

X = df.drop('Close/Last', axis=1)  # Input features (Date, Open, High, Low)
y = df['Close/Last']  # Target variable (Close/Last)

X = numpy.array(X)
y = numpy.array(y)'''

X = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x)

# define the size of the neural network
input_nodes = 2
hidden_nodes = 3
output_nodes = 1
batch_size = 8
batches = 1000
num_hidden_layers = 2

# defines input data and output data for testing and training model
input_data = X
output_data = y

# set weight arrays to random numbers (setting to 0 could cause a dead system)
weights = [numpy.random.randn(input_nodes, hidden_nodes)]
for _ in range(num_hidden_layers - 1):
  weights.append(numpy.random.randn(hidden_nodes, hidden_nodes))
weights.append(numpy.random.randn(hidden_nodes, output_nodes))

# set bias arrays to 0
biases = [numpy.zeros((1, hidden_nodes)) for _ in range(num_hidden_layers + 1)]

loss_array = numpy.array([[]])
indices = numpy.array([[]])
# loop for training iterations
for epoch in range(batches):
  # loop through hidden layers to add gradients
  for i in range(batch_size):
    # Extract data point from the batch
    current_input_data = input_data[epoch * batch_size + i]
    current_output_data = output_data[epoch * batch_size + i]
    
    # forward pass through all layers
    hidden_activations = [input_data]
    for layer in range(num_hidden_layers + 1):
      previous_activations = hidden_activations[-1]
      current_weights = weights[layer]
      hidden_values = previous_activations.dot(current_weights) + biases[layer]
      hidden_activations.append(np.maximum(0, hidden_values))  # ReLU
  
    # output predictions after last hidden layer
    output_data_predictions = hidden_activations[-1]
    
    gradients_weights = []
    gradients_biases = []

    output_error = output_data_predictions - current_output_data
    gradients_weights.append(previous_activations.T.dot(output_error))
    gradients_biases.append(output_error.sum(axis=0, keepdims=True))

    # backpropagate through hidden layers
    for layer in reversed(range(1, num_hidden_layers + 1)):
      previous_activations = hidden_activations[layer - 1]

      # ensures only activated neurons contribute to back propegation when finding error using relu
      layer_error = output_error.dot(weights[layer].T) * np.where(hidden_activations[layer] > 0, 1, 0)
      
      # update weights and biases
      gradients_weights.insert(0, previous_activations.T.dot(layer_error))
      gradients_biases.insert(0, layer_error.sum(axis=0, keepdims=True))

    # calculate loss using (guess - actual)^2 and add to array along with the index
    loss_array = numpy.append(loss_array, numpy.square(output_data_predictions - output_data).sum())
    indices = numpy.append(indices, i)
    
  # update weights and biases according to the slope found to be slightly closer to the minimum. 1e-4 is the learning rate
  for layer in range(num_hidden_layers + 1):
    weights[layer] -= gradient_weights[layer] * 1e-4
    biases[layer] -= gradient_biases[layer] * 1e-4

print("Output Data: ", output_data)
print("Output Predictions: ", output_data_predictions)
