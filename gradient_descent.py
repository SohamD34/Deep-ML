import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):

    n = len(y)
    curr_weights = weights

    if method == 'stochastic':
        for iter in range(n_iterations):
            for idx in range(n):
                y_pred = X[idx] @ curr_weights
                loss = (y_pred - y[idx]) ** 2
                gradient = 2 * (y_pred - y[idx]) * X[idx]
                curr_weights -= learning_rate * gradient

    elif method == 'mini_batch':
        for iter in range(n_iterations):
            indices = np.random.choice(n, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            y_pred = X_batch @ curr_weights
            loss = np.sum((y_pred - y_batch) ** 2) / batch_size
            gradient = (2 / batch_size) * (X_batch.T @ (y_pred - y_batch))
            curr_weights -= learning_rate * gradient

    else:  
        for iter in range(n_iterations):
            y_pred = X @ curr_weights
            loss = np.sum((y_pred - y) ** 2) / n
            gradient = (2 / n) * (X.T @ (y_pred - y))
            curr_weights -= learning_rate * gradient

    return curr_weights


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
y = np.array([2, 3, 4, 5])

# Parameters
learning_rate = 0.01
n_iterations = 1000
batch_size = 2

weights = np.zeros(X.shape[1])

final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
print(final_weights)

final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic')
print(final_weights)

final_weights = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch')
print(final_weights)