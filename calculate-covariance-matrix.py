import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

	return np.cov(vectors)