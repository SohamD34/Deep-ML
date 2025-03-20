import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

	# Validating invertibility - determinant != 0

	if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
		return -1

	transformed_matrix = np.linalg.inv(T) @ A @ S

	return transformed_matrix