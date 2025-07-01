def compressed_col_sparse_matrix(dense_matrix):

    if not dense_matrix or not dense_matrix[0]:
        return [], [], []

    num_rows = len(dense_matrix)
    num_cols = len(dense_matrix[0])
    
    vals = []
    row_idx = []
    col_ptr = [0] * (num_cols + 1)

    for j in range(num_cols):
        for i in range(num_rows):
            if dense_matrix[i][j] != 0:
                vals.append(dense_matrix[i][j])
                row_idx.append(i)
                col_ptr[j + 1] += 1

    # Convert col_ptr to cumulative sum
    for j in range(1, len(col_ptr)):
        col_ptr[j] += col_ptr[j - 1]

    return vals, row_idx, col_ptr
	


dense_matrix = [
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
]

vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
print(vals)
print(row_idx)
print(col_ptr)