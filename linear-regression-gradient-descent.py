import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	
	m, n = X.shape
	theta = np.zeros((n, ))

	for it in range(iterations):
		pred = np.dot(X, theta)
		error = pred - y
		theta = theta - (alpha/m) * np.dot(X.T, error)

	return theta