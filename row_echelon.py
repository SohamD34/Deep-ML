# Task: Compute the Column Space of a Matrix

# In this task, you are required to implement a function matrix_image(A) that calculates the column space of a given matrix A. 
# The column space, also known as the image or span, consists of all linear combinations of the columns of A. 
# To find this, you'll use concepts from linear algebra, focusing on identifying independent columns that span the matrix's image. 
# Your task: Implement the function matrix_image(A) to return the basis vectors that span the column space of A. 
# These vectors should be extracted from the original matrix and correspond to the independent columns.
# Example:

# Input:
# matrix = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# print(matrix_image(matrix))

# Output:
# [[1, 2],
#  [4, 5],
#  [7, 8]]


import numpy as np

def matrix_image(A):
    """    
    1. Convert to Row Echelon Form (RREF)
    The first step is to convert the matrix to its RREF using Gauss-Jordan Elimination. This finds the independent equations within the matrix. 
    In RREF form:
    - Each non-zero row begins with a leading 1, called a pivot
    - Rows of all zeros are at the bottom of the matrix
    - Each leading 1 is to the right of the leading 1 in the row above

    2. Identify Pivot Columns
    Once the matrix is in RREF, the pivot columns are the columns that contain the leading 1s in each non-zero row. These columns represent the independent directions that span the column space of the matrix.

    3. Extract Pivot Columns from the Original Matrix
    Finally, to find the column space of the original matrix, you take the columns from the original matrix corresponding to the pivot columns in RREF.
    """

    A_rref = np.array(A, dtype=float)  
    rows, cols = A_rref.shape
    pivot_columns = []

    for r in range(rows):
        for c in range(cols):
            if A_rref[r, c] != 0:
                pivot_columns.append(c)
                A_rref[r] = A_rref[r] / A_rref[r, c]
                for rr in range(rows):
                    if rr != r:
                        A_rref[rr] -= A_rref[rr, c] * A_rref[r]
                break

    independent_columns = A[:, pivot_columns]

    return independent_columns


matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(matrix))