import numpy as np
MAXN = 100

def partial_pivot(A, n):
    for i in range(n):
        pivot_row = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[pivot_row][i]):
                pivot_row = j
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor * A[i]


def back_substitute(A, n):
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (A[i][n] - sum_val) / A[i][i]
    return x


def gaussian_elimination(A, b):
    n = len(b)
	
    A_b = []
    for i in range(len(A)):
        A_b.append(list(np.append(A[i], b[i])))

    A_b = np.array(A_b, dtype=float)
    partial_pivot(A_b, n)
    x = back_substitute(A_b, n)
    return x


A = np.array([[2,8,4], [2,5,1], [4,10,-1]], dtype=float)
b = np.array([2,5,1], dtype=float)
print(gaussian_elimination(A, b))