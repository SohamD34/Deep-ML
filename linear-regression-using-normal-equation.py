import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	
	# Normal Equations
	# theta = (XTX)-1.(XTY)
	
	X_T = np.transpose(X)

	theta = np.dot(np.linalg.inv(X_T @ X), (X_T @ y))

	return theta