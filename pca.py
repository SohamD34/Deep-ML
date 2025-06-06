import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:

    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    std_data = (data - mean) / std_dev

    cov_mat = np.cov(std_data, rowvar=False)
    ev, evec = np.linalg.eig(cov_mat)

    sorted_indices = np.argsort(ev)[::-1]
    sorted_evec = evec[:, sorted_indices]
    principal_components = sorted_evec[:, :k]

    return np.round(principal_components, 4)


data = np.array([[1, 2], [3, 4], [5, 6]])
k = 1
eig_vec = pca(data, k)

print("Principal Components:\n", eig_vec)