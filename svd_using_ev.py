import numpy as np

def svd_2x2(A: np.ndarray):

    assert A.shape == (2, 2), "Input must be a 2x2 matrix"

    ATA = A.T @ A

    eigvals, V = np.linalg.eigh(ATA)  

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    singular_values = np.sqrt(np.clip(eigvals, 0, None))  

    U = np.zeros((2, 2))
    for i in range(2):
        if singular_values[i] > 1e-10:  
            U[:, i] = A @ V[:, i] / singular_values[i]
        else:
            U[:, i] = np.array([-U[1, 0], U[0, 0]])

    S = np.diag(singular_values)

    assert A.all() == (U @ S @ V.T).all(), "Reconstruction check failed"

    return U, S, V.T


A = np.array([[-10, 8], [10, -1]])
U, S, V = svd_2x2(A)

# print("U:\n", U)
# print("S:\n", S)
# print("V^T:\n", V)

print(U @ np.diag(S) @ V)
