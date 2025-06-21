import numpy as np

def r_squared(y_true, y_pred):
	
    errors = [y_true[i] - y_pred[i] for i in range(len(y_true))]

    mean = np.mean(y_true)
    dev = [i - mean for i in y_true]

    SSR = np.sum([i**2 for i in errors])
    SST = np.sum([i**2 for i in dev])

    return 1 - SSR/SST
