import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Convert input vectors to numpy arrays
    vectors = [np.array(v, dtype=float) for v in vectors]
    
    w = []
    w.append(vectors[0])

    for i in range(1, len(vectors)):
        w_i = vectors[i].copy()
        for j in range(i):
            proj = np.dot(vectors[i], w[j]) / np.dot(w[j], w[j]) * w[j]
            w_i = w_i - proj
        w.append(w_i)

    u = []
    for w_i in w:
        norm = np.linalg.norm(w_i)
        if norm > tol:
            u_i = w_i / norm
            u.append(u_i)

    return u

print(orthonormal_basis([[1, 0], [1, 1]]))