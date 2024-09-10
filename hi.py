import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        return self.sigmoid(x) * (1 - self.sigmoid(X))

    def forward(self, X):
        # Forward propagation
        hidden_layer_input = np.dot(X, self.weights1) + self.bias1
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights2) + self.bias2
        predicted_output = self.sigmoid(output_layer_input)
        return predicted_output

    def backward(self, X, y, predicted_output, learning_rate):
        # Backward propagation
        error = y - predicted_output
        d_predicted_output = error * self.sigmoid_derivative(predicted_output)

        d_weights2 = np.dot(hidden_layer_output.T, d_predicted_output)
        d_bias2 = np.sum(d_predicted_output, axis=0)

        d_hidden_layer_output = np.dot(d_predicted_output, self.weights2.T)
        d_hidden_layer_input = d_hidden_layer_output * self.sigmoid_derivative(hidden_layer_output)

        d_weights1 = np.dot(X.T, d_hidden_layer_input)
        d_bias1 = np.sum(d_hidden_layer_input, axis=0)

        # Update weights and biases
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
      
    def train(self, X_train, y_train, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
          predicted_output = self.forward(X_train)
          loss = self.calculate_loss(predicted_output, y_train)
          self.backward(X_train, y_train, predicted_output, learning_rate)
          
        return loss

nn = NeuralNetwork(2, 3, 1)  # 2 inputs, 3 neurons in hidden layer, 1 output
nn.train(X_train, y_train)

# Test the network
y_pred = nn.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_labels)
print("Accuracy:", accuracy)
