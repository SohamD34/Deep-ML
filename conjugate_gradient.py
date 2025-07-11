# Task: Implement the Conjugate Gradient Method for Solving Linear Systems
# Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative algorithm for solving large, sparse, symmetric, positive-definite linear systems. Given a matrix A and a vector b, the algorithm will solve for x in the system ( Ax = b ).
# Write a function conjugate_gradient(A, b, n, x0=None, tol=1e-8) that performs the Conjugate Gradient method as follows:

#     A: A symmetric, positive-definite matrix representing the linear system.
#     b: The vector on the right side of the equation.
#     n: Maximum number of iterations.
#     x0: Initial guess for the solution vector.
#     tol: Tolerance for stopping criteria.

# The function should return the solution vector x.


import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x


    f(x) = 0.5 * (x.T @ A @ x) - (b.T @ x)
    f'(x) = Ax - b


    """
    x = np.zeros_like(b)
    if x0:
        x = x0
    r = b - A @ x           # residual vector (r)
    p = r                   # search direction (p)

    for iter in range(n):

        alpha = (r.T @ r)/(p.T @ A @ p)     # step size
        x = x + alpha*p                     # update current solution
        r_new = r - alpha*(A @ p)           # update residual

        if np.linalg.norm(r_new) < tol:
            break
        
        beta = (r_new.T @ r_new)/(r.T @ r)      # direction scaling - to ensure search is orthogonal to A
        p_new = r_new + beta*p                  # update direction
        p = p_new
        r = r_new
    
    return x


A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
n = 5

print(conjugate_gradient(A, b, n))