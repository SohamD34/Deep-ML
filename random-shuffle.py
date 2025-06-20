import numpy as np

def shuffle_data(X, y, seed=None):
	
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]

X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8]])
y = np.array([1, 2, 3, 4])
X_shuffled, y_shuffled = shuffle_data(X, y, seed=42)
print("Shuffled X:\n", X_shuffled)
print("Shuffled y:\n", y_shuffled)