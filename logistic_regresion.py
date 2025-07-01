import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    threshold = lambda x: 1 if x >= 0.5 else 0

    y = np.dot(X, weights) + bias
    y = sigmoid(y)
    y = np.vectorize(threshold)(y)
    return y


# print(predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0))
print(predict_logistic(np.array([[0, 0], [0.1, 0.1], [-0.1, -0.1]]), np.array([1, 1]), 0))