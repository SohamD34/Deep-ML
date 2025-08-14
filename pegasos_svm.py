import numpy as np

"""
PEGASOS (Primal Estimated sub-GrAdient SOlver for SVM)  

 - Fast, iterative algorithm designed to train Support Vector Machines (SVM).
 - Uses stochastic updates over every data sample

// Kernel functions - 
 Linear                      -- K(x,y) = x.y
 Radial basis function (RBF) -- K(x,y) = exp( -(norm(x-y))**2 / (2 * sigma**2))

// Regularization param - Lambda

// Sub-gradient descent - sub-gradient of Hinge loss


ALGORITHM :-

1) Initialize alphas = 0, bias b = 0.
2) Iterate:
    a) Compute LR.              n(t) = 1/lambda*t
    b) For each sample (xi,yi):
        - Decision value        f(xi) = SUM(j) [alphaj * yj * K(xj, xi)] + b
        - If yi*f(xi) < 1:
            alphai = alphai + n(t)*(yi - lambda*alphai)
            b = b + n(t)*yi

"""


def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    
            decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return np.round(alphas, 4).tolist(), np.round(b, 4)




data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]])
labels = np.array([1, 1, -1, -1])
kernel = 'rbf'
lambda_val = 0.01 
iterations = 100 
sigma = 1.0
alphas, b = pegasos_kernel_svm(data, labels, kernel, lambda_val, iterations, sigma)
print(alphas)
print(b)