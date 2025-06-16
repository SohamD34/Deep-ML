import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    
    batches = []
    current_idx = 0

    while current_idx < len(X):
        # print(current_idx)

        if current_idx + batch_size <= len(X):
            batch_X = X[current_idx: current_idx + batch_size]

            if y is not None:
                batch_y = y[current_idx: current_idx + batch_size]
            else:
                batch_y = None
            current_idx += batch_size
        else:
            batch_X = X[current_idx: len(X)]

            if y is not None:
                batch_y = y[current_idx: len(X)]
            else:
                batch_y = None
            current_idx = len(X)

        if batch_y is not None:
            batches.append([batch_X, batch_y])
        else:
            batches.append([batch_X])

    return batches


# X = np.array([[1, 2], 
#               [3, 4], 
#               [5, 6], 
#               [7, 8], 
#               [9, 10]])
# y = np.array([1, 2, 3, 4, 5])
# batch_size = 2
# print(batch_iterator(X, y, batch_size))

print(batch_iterator(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), batch_size=3))