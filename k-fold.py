import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    
    n = len(X)
    test_size = n // k

    splits = []

    if shuffle:
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    for fold in range(k):
        test_indices_start = fold * test_size
        test_indices_end = test_indices_start + test_size

        test_indices = indices[test_indices_start:test_indices_end]
        train_indices = np.concatenate((indices[:test_indices_start], indices[test_indices_end:]))

        splits.append((train_indices.tolist(), test_indices.tolist()))

    return splits


print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False))
# Expected - [([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]

print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=True))
# Expected - [([2, 9, 4, 3, 6], [8, 1, 5, 0, 7]), ([8, 1, 5, 0, 7], [2, 9, 4, 3, 6])]
