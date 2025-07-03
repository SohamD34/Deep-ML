import numpy as np

def cramers_rule(A, b):
    
    A = np.array(A)
    b = np.array(b)
    n = A.shape[0]
    x = []

    A_det = np.linalg.det(A) 
    
    if A_det== 0:
        return -1

    for i in range(n):
        A_i = A.copy()
        for j in range(n):
            A_i[j][i] = b[j]

        x_i = np.linalg.det(A_i) / np.linalg.det(A)
        x.append(round(x_i, 4))
    
    return x


A = [[2, -1, 3], [4, 2, 1], [-6, 1, -2]]
b = [5, 10, -3]
print(cramers_rule(A, b))