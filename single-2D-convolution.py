import numpy as np

def print_2d_matrix(matrix):
	print()
	for row in matrix:
		print(row)
	print()
	

def add_padding(matrix, pad):
	m, n = matrix.shape
	new_rows, new_cols = m + 2 * pad, n + 2 * pad
	new_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)
	new_matrix[pad:pad+m, pad:pad+n] = matrix
	return new_matrix



def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	
	hin, win = input_matrix.shape
	hk, wk = kernel.shape
	hout = (hin + 2*padding - hk)//stride + 1
	wout = (win + 2*padding - wk)//stride + 1

	out_matrix = np.zeros((hout, wout))

	padded_matrix = add_padding(input_matrix, padding)

	for r in range(hout):
		for c in range(wout):

			out_matrix[r, c] = np.sum(
				padded_matrix[r*stride:r*stride+hk, c*stride:c*stride+wk] * kernel
			)

	return out_matrix


input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print_2d_matrix(output)