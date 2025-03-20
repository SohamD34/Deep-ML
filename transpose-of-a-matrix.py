import numpy as np

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:

	b = (np.array(a).T).tolist()
	return b