import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

	if np.linalg.det(matrix) == 0:
		return None

	inverse = np.linalg.inv(matrix)
	return inverse