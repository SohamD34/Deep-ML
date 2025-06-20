import numpy as np

def rmse(y_true, y_pred):
    return round(np.sqrt(np.mean((y_true - y_pred).ravel() ** 2)), 3)

y_true2 = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred2 = np.array([[0, 2], [-1, 2], [8, -5]])
print(rmse(y_true2, y_pred2))
# Expected output: 0.842