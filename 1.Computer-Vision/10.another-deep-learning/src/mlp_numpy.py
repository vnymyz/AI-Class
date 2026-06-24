"""
Tiny MLP built from scratch with raw numpy.
No autograd, no framework — forward pass, backward pass (chain rule),
and weight updates are all written by hand so you see exactly what's
happening at each step.

Architecture: 2 inputs -> hidden layer (sigmoid) -> 1 output (sigmoid)
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(activated):
    # derivative of sigmoid, expressed in terms of its own output
    return activated * (1 - activated)


class NumpyMLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, seed=42):
        rng = np.random.default_rng(seed)
        # small random init so neurons don't all start identical
        self.W1 = rng.standard_normal((input_size, hidden_size)) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)          # hidden layer output
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)          # final prediction
        return self.a2

    def backward(self, X, y, prediction, learning_rate):
        n_samples = X.shape[0]

        # --- step 1: how wrong are we, and which direction reduces it? ---
        # derivative of MSE loss w.r.t. prediction, times sigmoid derivative
        error_output = (prediction - y) * sigmoid_derivative(prediction)

        # --- step 2: gradients for output layer weights/bias ---
        dW2 = self.a1.T @ error_output / n_samples
        db2 = np.sum(error_output, axis=0, keepdims=True) / n_samples

        # --- step 3: push the "blame" backward into the hidden layer ---
        error_hidden = (error_output @ self.W2.T) * sigmoid_derivative(self.a1)

        # --- step 4: gradients for hidden layer weights/bias ---
        dW1 = X.T @ error_hidden / n_samples
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / n_samples

        # --- step 5: nudge every weight a small step downhill ---
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def compute_loss(self, prediction, y):
        return np.mean((prediction - y) ** 2)  # MSE

    def train_step(self, X, y, learning_rate):
        prediction = self.forward(X)
        loss = self.compute_loss(prediction, y)
        self.backward(X, y, prediction, learning_rate)
        return loss
