import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-15  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivation(z):
    return sigmoid(z) * (1 - sigmoid(z))


def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    
    weights = np.zeros(X.shape[1])
    bias = 0.0
    all_losses = []

    for iter in range(iterations):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        loss = round(binary_cross_entropy(y, y_pred), 4)
        all_losses.append(loss)

        dw = np.dot(X.T, (y_pred - y)) / y.size
        db = np.mean(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    parameters = np.round(weights, 4).tolist() + [round(bias, 4)]
    
    return (parameters, all_losses)

