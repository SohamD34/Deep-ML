import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	
	a = np.array(a)
	b = np.array(b)

	(x1, y1) = a.shape
	(x2, y2) = b.shape

	if(y1 != x2):
		return -1

	c = a @ b
	return c