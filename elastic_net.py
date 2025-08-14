import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    
    n = len(X)
    W = np.zeros(X.shape[1])
    b = 0

    for iter in range(max_iter):
        y_pred = np.dot(X, W) + b
        error = y_pred - y

        grad_W = np.dot(X.T, error)/n + alpha1*np.sign(W) + 2*alpha2*W
        grad_b = np.sum(error)/n

        W = W - learning_rate*grad_W
        b = b - learning_rate*grad_b

        # convergence
        l1_norm = np.linalg.norm(grad_W)
        if l1_norm < tol:
            break
    
    W = np.round(W, 2)
    b = np.round(b, 2)
    return W, b


X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])
print(elastic_net_gradient_descent(X, y))