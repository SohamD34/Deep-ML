import numpy as np

def sigmoid(x):
    return  1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return np.array([sig_x*(1 - sig_x) for sig_x in sigmoid_x])

def mse(y_true, y_pred):
    n = len(y_true)
    s = 0.0
    for i in range(n):
        s += (y_true[i] - y_pred[i])**2
    return s/n


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	
    X = features
    y_true = labels
    W = initial_weights
    b = initial_bias
    lr = learning_rate
    n = len(X)

    mse_values = []

    for e in range(epochs):

        z = X @ W + b
        y = sigmoid(z)

        loss = round(mse(y, labels), 4)
        mse_values.append(loss)

        '''
        L = 1/n * sum(y_true - sigmoid(z))**2
        dL/dz = -1/n * sum(2 * (y_true - sigmoid(z)) * sigmoid_der(z))

        z = X@W + b
        dz/dw = 

        '''

        dL_dz = -(2/n) * (y_true - y) * sigmoid_derivative(z)  
        dL_dw = np.dot(X.T, dL_dz) 
        dL_db = np.sum(dL_dz)  

        W -= lr * dL_dw  
        b -= lr * dL_db  

    return np.round(W, 4), np.round(b, 4), mse_values


features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
labels = np.array([1, 0, 0])
initial_weights = np.array([0.1, -0.2])
initial_bias = 0.0
learning_rate = 0.1
epochs = 2
print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))