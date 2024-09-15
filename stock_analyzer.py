import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ReLU(input):
    return np.maximum(0, input)
    
def MSE(y_true, y_pred):
    return (y_true - y_pred) ** 2
    
def linearize(input, weight, bias):
    return numpy.dot(input, weight) + bias
    
def gradient_descent(learning_rate, weights, biases, gradients):
    for layer in range(len(weights)):
        weights[layer] -= learning_rate * gradients[layer]
    for layer in range(len(biases)):
        biases[layer] -= learning_rate * gradients[layer + 1]

# define the size of the neural network
input_nodes = 5
hidden_nodes = 4
output_nodes = 1
batch_size = 32
batches = 1000
num_hidden_layers = 2
learning_rate = 1e-3

# import data from past 5 years of nasdaq
data_frame = pd.read_csv('5year_stock.csv')
X_date = df['Date'].values[:-1] # excludes last point for current stock
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]
y = df['Close'].values[1:] # excludes first point for tomorrow stock

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(np.stack([X_date, X_close, X_open, X_high, X_low], axis=1))

# set weight arrays to random numbers (setting to 0 could cause a dead system)
weights = [np.random.randn(input_nodes, hidden_nodes)]
for _ in range(num_hidden_layers - 1):
  weights.append(np.random.randn(hidden_nodes, hidden_nodes))
weights.append(np.random.randn(hidden_nodes, output_nodes))

# set bias arrays to 0
biases = [np.zeros((1, hidden_nodes)) for _ in range(num_hidden_layers + 1)]

# Training loop
for epoch in range(batches):
    num_correct = 0
    for i in range(batch_size):        
        current_X = X[epoch * batch_size + i]
        current_y = y[epoch * batch_size + i]
        
        # calculate output guess
        y_pred = ReLU(linearize(X, weights[0], biases[0]))
        hidden_values = []
        for layer in range(num_hidden_layers):
            hidden_values.append(y_pred)
            y_pred = ReLU(linearize(y_pred, weights[layer+1], biases[layer+1]))

        # add current loss to losses
        loss = MSE(current_y, y_pred)
        num_correct += int(np.argmax(y_pred) == np.argmax(y))
        grad_pred = 2 * (y_pred - current_y)

        # find gradient weights and biases
        grad_weights = []
        grad_biases = []
        for i in range(len(weights) - 1, 0, -1):
            grad_hidden = np.dot(grad_pred, weights[i].T) * (hidden_values[i - 1] > 0)
            grad_weights.append(np.dot(hidden_values[i - 1].T, grad_hidden))
            grad_biases.append(np.sum(grad_hidden, axis=0, keepdims=True))
        
        # Update weights and biases
        gradient_descent(learning_rate, weights, biases, gradients)
    
    print("Accuracy: ", num_correct / batch_size)

print("Output Data: ", y)
print("Output Predictions: ", y_pred)
