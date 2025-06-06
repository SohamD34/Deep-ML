import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	
    n = len(X)
    y_pred = np.dot(X, w)
    mse = 0.0

    for i in range(n):
        mse += (y_pred[i] - y_true[i])**2
    mse /= n

    reg = np.sum([w_i**2 for w_i in w])
    ridge = mse + alpha*reg

    return ridge


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
w = np.array([0.2, 2])
y_true = np.array([2, 3, 4, 5])
alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)
print(loss)