import numpy as np

def rref(A):
    A_rref = np.array(A, dtype=float)
    rows, cols = A_rref.shape
    row = 0
    
    for col in range(cols):
        if row >= rows:
            break

        pivot = None
        for r in range(row, rows):
            if A_rref[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        
        if pivot != row:
            A_rref[[row, pivot]] = A_rref[[pivot, row]]

        A_rref[row] = A_rref[row] / A_rref[row, col]

        for r in range(rows):
            if r != row:
                A_rref[r] -= A_rref[r, col] * A_rref[row]
        row += 1
    return A_rref