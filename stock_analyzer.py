'''
Purpose: Analyze stock data using a neural network

Owen Colley
9/14/24
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def MSE(y_pred, y):
    return (y - y_pred) ** 2
    
# import data from past 5 years of nasdaq
df = pd.read_csv('5year_stock.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
X_day = df['Date'].dt.day.values[:-1] # excludes last point for current stock
X_month = df['Date'].dt.month.values[:-1]
X_year = df['Date'].dt.year.values[:-1]
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]

# normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(np.column_stack([X_month, X_close, X_high, X_low]))
y = scaler.fit_transform(df['Close'].values[1:].reshape(-1, 1)) # excludes first point for tomorrow stock

# define the size of the neural network
input_nodes = 4
hidden_nodes = 6
output_nodes = 1
batch_size = 1
batches = 1000
num_hidden_layers = 1
learning_rate = 1e-4

# set weight arrays to random numbers (setting to 0 could cause a dead system)
weights = [np.random.randn(input_nodes, hidden_nodes)]
for _ in range(num_hidden_layers-1):
    weights.append([np.random.randn(hidden_nodes, hidden_nodes)])
weights.append(np.random.randn(hidden_nodes, output_nodes))

# set bias arrays to 0
biases = []
for _ in range(num_hidden_layers):
    biases.append([np.zeros(hidden_nodes)])
biases.append(np.zeros(output_nodes))

# training loop
for epoch in range(batches):
    start = epoch * batch_size
    current_X = X[start : start + batch_size]
    current_y = y[start : start + batch_size]
    
    # pass through hidden layers and output
    y_pred = current_X
    pre_activations = []
    for layer in range(len(weights)):
        z = np.dot(y_pred, weights[layer]) + biases[layer]
        pre_activations.append(z)
        y_pred = np.maximum(0, z)
    
    # set gradient prediction to derivative of loss function
    grad_pred = 2 * (y_pred - current_y)

    # find gradient weights and biases
    grad_weights = []
    grad_biases = []
    for layer in reversed(range(len(weights))):
        # update weight and bias gradient arrays
        grad_weights.append(np.dot(pre_activations[layer], grad_pred.T))
        grad_biases.append(np.sum(grad_pred, axis=0))

        # update grad_pred for next layer because chain rule
        if layer != 0:
            grad_pred = np.dot(grad_pred.T, weights[layer].T)
        
    # update weights and biases with gradient descent
    for layer in range(len(weights)):
        weights[layer] -= learning_rate * grad_weights[len(weights) - 1 - layer]
    for layer in range(len(biases)):
        biases[layer] -= learning_rate * grad_biases[len(biases) - 1 - layer]

test_preds = []
X_test = X[20:30]
y_test = y[20:30]
for input in X_test:
    y_pred = input
    for layer in range(len(weights)):
        z = np.dot(y_pred, weights[layer]) + biases[layer]
        y_pred = ReLU(z)
    test_preds.append(y_pred)

mse_test = np.mean(MSE(test_preds, y_test))
print("mean squared error:", mse_test)
print(f"test data: {X[20][1]} result: {test_preds[0][0][0]}")
