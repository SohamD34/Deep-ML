import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	
	means = []

	if mode == 'column':
		means = np.mean(matrix, axis=0)
	else:
		means = np.mean(matrix, axis=1)

	means = means.tolist()
	return means