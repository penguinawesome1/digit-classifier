import pandas as pd
import numpy as np

data_frame = pd.read_csv('5year_stock.csv')
X_date = df['Date'].values[:-1]
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]

y = df['Close'].values[1:]

def ReLU(input):
    return max(0, input)
def softmax(input):
    exp_input = np.exp(input)
    return exp_input / np.sum(exp_input, axis=-1, keepdims=True)
def MSE(guess, actual):
    return ((actual - guess) ** 2).mean()
def gradient_descent(learning_rate, weights, biases, gradients):
    weights -= learning_rate * gradients["weights"]
    biases -= learning_rate * gradients["biases"]
def step_function(x):
    return 1 if x >= 0 else 0
def linear_function(input, weight, bias):
    return input * weight + bias
