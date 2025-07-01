import numpy as np


def calculate_correlation_matrix(X, Y=None):

    if Y is None:
        Y = X
    
    n_samples, n_features_X = X.shape
    n_features_Y = Y.shape[1]
    
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    X_std = np.std(X, axis=0, ddof=0)
    Y_std = np.std(Y, axis=0, ddof=0)
    
    correlation_matrix = np.zeros((n_features_X, n_features_Y))
    
    for i in range(n_features_X):
        for j in range(n_features_Y):
            if X_std[i] == 0 or Y_std[j] == 0:
                if X_std[i] == 0 and Y_std[j] == 0:
                    if np.allclose(X[:, i], Y[:, j]):
                        correlation_matrix[i, j] = 1.0
                    else:
                        correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
            else:
                covariance = np.mean(X_centered[:, i] * Y_centered[:, j])
                correlation_matrix[i, j] = covariance / (X_std[i] * Y_std[j])
    
    return correlation_matrix


print(calculate_correlation_matrix(np.array([[1, 0], [0, 1]]), np.array([[1, 2], [3, 4]])))
# Expected
# [[ -1., -1.], [ 1., 1.]]