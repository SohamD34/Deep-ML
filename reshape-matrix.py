import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	
	x = len(a)
	y = len(a[0])

	if x*y != new_shape[0]*new_shape[1]:
		return []
		
	reshaped_matrix = np.array(a).reshape(new_shape).tolist()
	return reshaped_matrix