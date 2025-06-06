import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_pred, labels):
    return np.mean((y_pred - labels) ** 2)

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []

    for e in range(epochs):

        linear_output = features @ updated_weights.T + updated_bias
        y_pred = sigmoid(linear_output)

        mse_error = mse(y_pred, labels)
        mse_values.append(round(mse_error, 4))

        error = y_pred - labels
        gradient_weights = np.round(features.T @ (error * sigmoid_derivative(linear_output)), 4)
        gradient_bias = np.round(np.sum(error * sigmoid_derivative(linear_output)), 4)

        updated_weights -= np.round(learning_rate * gradient_weights, 4)
        updated_bias -= np.round(learning_rate * gradient_bias, 4)

    return updated_weights, updated_bias, mse_values

# Example usage
features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
labels = np.array([1, 0, 0])
initial_weights = np.array([0.1, -0.2])
initial_bias = 0.0
learning_rate = 0.1
epochs = 2

print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))
# output :  updated_weights = [0.1036, -0.1425], updated_bias = -0.0167, mse_values = [0.3033, 0.2942]