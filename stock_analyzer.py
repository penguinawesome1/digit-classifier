import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ReLU(input):
    return np.maximum(0, input)
def softmax(input):
    exp_input = np.exp(input)
    return exp_input / np.sum(exp_input, axis=-1, keepdims=True)
def MSE(y_true, y_pred):
    return (y_true - y_pred) ** 2
def tanh(x):
    return np.tanh(x)
def gradient_descent(learning_rate, weights, biases, gradients):
    weights -= learning_rate * gradients["weights"]
    biases -= learning_rate * gradients["biases"]
def step_function(x):
    return 1 if x >= 0 else 0
def linear_function(input, weight, bias):
    return input * weight + bias

# import data from past 5 years of nasdaq
data_frame = pd.read_csv('5year_stock.csv')
X_date = df['Date'].values[:-1]
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]
y = df['Close'].values[1:]

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(np.stack([X_date, X_close, X_open, X_high, X_low], axis=1))

# define the size of the neural network
input_nodes = 5
hidden_nodes = 4
output_nodes = 1
batch_size = 32
batches = 1000
num_hidden_layers = 2

# set weight arrays to random numbers (setting to 0 could cause a dead system)
weights = [np.random.randn(input_nodes, hidden_nodes)]
for _ in range(num_hidden_layers - 1):
  weights.append(np.random.randn(hidden_nodes, hidden_nodes))
weights.append(np.random.randn(hidden_nodes, output_nodes))

# set bias arrays to 0
biases = [np.zeros((1, hidden_nodes)) for _ in range(num_hidden_layers + 1)]

losses = np.array([[]])

# Training loop
for epoch in range(batches):
    for i in range(batch_size):
        # Forward pass
        current_input = X[epoch * batch_size + i]
        current_output = y[epoch * batch_size + i]

        # Calculate activations for hidden layers
        for layer in range(num_hidden_layers):
            z = np.dot(current_input, weights[layer]) + biases[layer]
            current_input = tanh(z)

        # Calculate output
        output = np.dot(current_input, weights[-1]) + biases[-1]

        # Calculate error and gradients
        error = MSE(current_output, y_pred)
        # ... (backpropagation to calculate gradients)

        # Update weights and biases
        for layer in range(num_hidden_layers + 1):
            weights[layer] -= learning_rate * gradients[layer]
            biases[layer] -= learning_rate * gradients[layer + 1]

# Generate predictions
y_predictions = []
for x in X:
    # Forward pass
    for layer in range(num_hidden_layers):
        z = np.dot(x, weights[layer]) + biases[layer]
        x = tanh(z)
    output = np.dot(x, weights[-1]) + biases[-1]
    y_predictions.append(output)

print("Output Data: ", y)
print("Output Predictions: ", y_predictions)
