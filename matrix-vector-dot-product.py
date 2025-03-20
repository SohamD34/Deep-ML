import numpy as np


def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
	
	# assuming that shape of the matrix a = (m x n)
	m = len(a)
	n = len(a[0])
	
	if m!=len(b):
		return -1

	c = [np.dot(i, b) for i in a]

	return c