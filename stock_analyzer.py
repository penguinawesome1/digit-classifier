'''
Purpose: Analyze stock data using a neural network

Owen Colley
9/14/24
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ReLU(input):
    return np.maximum(0, input)
    
def MSE(y_pred, y):
    return (y - y_pred) ** 2
    
def linearize(input, weight, bias):
    return np.dot(input, weight) + bias
    
def gradient_descent(learning_rate, weights, biases, grad_weights, grad_biases):
    for layer in range(len(weights)):
        weights[layer] -= learning_rate * grad_weights[layer]
    for layer in range(len(biases)):
        biases[layer] -= learning_rate * grad_biases[layer]

# define the size of the neural network
input_nodes = 5
hidden_nodes = 8
output_nodes = 1
batch_size = 64
batches = 10
num_hidden_layers = 3
learning_rate = 1e-3

# import data from past 5 years of nasdaq
df = pd.read_csv('5year_stock.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
X_month = df['Date'].dt.month.values[:-1] # excludes last point for current stock
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]
y = df['Close'].values[1:] # excludes first point for tomorrow stock

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(np.column_stack([X_month, X_close, X_open, X_high, X_low]))

# set weight arrays to random numbers (setting to 0 could cause a dead system)
weights = [np.random.randn(input_nodes, hidden_nodes)]
for _ in range(num_hidden_layers - 1):
  weights.append(np.random.randn(hidden_nodes, hidden_nodes))
weights.append(np.random.randn(hidden_nodes, output_nodes))

# set bias arrays to 0
biases = [np.zeros((1, hidden_nodes)) for _ in range(num_hidden_layers)]
biases.append(np.zeros((1, output_nodes)))

# Training loop
for epoch in range(batches):
    for i in range(batch_size):
        start = i * batch_size
        current_X = X[start : start + batch_size]
        current_y = y[start : start + batch_size]
        
        # calculate output guess
        y_pred = current_X.reshape(-1, input_nodes)
        pre_activations = []
        hidden_values = []
        for layer in range(num_hidden_layers):
            z = linearize(y_pred, weights[layer], biases[layer])
            pre_activations.append(z)
            y_pred = ReLU(z)
            hidden_values.append(y_pred)
        y_pred = ReLU(linearize(y_pred, weights[num_hidden_layers], biases[num_hidden_layers]))
        
        # set gradient prediction to derivative of loss function
        grad_pred = 2 * (y_pred.T - current_y)

        # find gradient weights and biases
        grad_weights = []
        grad_biases = []
        for layer in reversed(range(num_hidden_layers)):
            grad_weights.append(np.dot(pre_activations[layer].T, grad_pred.T))
            grad_biases.append(np.sum(grad_pred, axis=0, keepdims=True))
        
    # Update weights and biases
    gradient_descent(learning_rate, weights, biases, grad_weights, grad_biases)

print("Output Data: ", y)
print("Output Predictions: ", y_pred)
