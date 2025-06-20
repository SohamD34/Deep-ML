import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):

    if len(y)%n_subsets == 0:
        size = len(y)// n_subsets
    else:
        size = len(y) // n_subsets + 1

    if seed is not None:
        np.random.seed(seed)

    subsets = []
    for _ in range(n_subsets):
        if replacements:
            indices = np.random.choice(len(y), size=size, replace=True)
        else:
            indices = np.random.choice(len(y), size=size, replace=False)
        subsets.append((X[indices], y[indices]))
    return subsets


X = np.array([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10]])
y = np.array([1, 2, 3, 4, 5])
n_subsets = 3
replacements = False
print(get_random_subsets(X, y, n_subsets, replacements))