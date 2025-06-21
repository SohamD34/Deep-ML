import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    out = []
    for x in X:
        x = np.append(1, x)
        comb = combinations_with_replacement(x, degree)
        
        prods = []
        for tuples in list(comb):
            prod = 1
            for num in list(tuples):
                prod *= num
            prods.append(prod)
        out.append(sorted(prods))
    return out



# X = np.array([[2, 3],
#               [3, 4],
#               [5, 6]])
# degree = 2
# output = polynomial_features(X, degree)
# print(output)

print(polynomial_features(np.array([[1, 2], [3, 4], [5, 6]]), 3))