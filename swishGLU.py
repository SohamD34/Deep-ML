import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def swish(x):
    return x*sigmoid(x)


def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    batch_size, d2 = x.shape
    d = d2 // 2  # Use integer division for slicing

    x1 = x[:, :d]
    x2 = x[:, d:]
    
    return x1 * swish(x2)


print(SwiGLU(np.array([[1, -1, 1000, -1000]])))