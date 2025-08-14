import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    
    n = len(y)
    
    for _ in range(n_iterations):

        if method == 'batch':
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient
        
        elif method == 'stochastic':
            for i in range(n):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient
        
        elif method == 'mini_batch':
            for i in range(0, n, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient
                
    return weights


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