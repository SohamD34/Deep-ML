import numpy as np 

def svd_2x2_singular_values(A: np.ndarray) -> tuple:

    B = A.T @ A
    optimal_theta = None

    if B[0,0] == B[1,1]:
            optimal_theta = np.pi / 4  # 45 degrees in radians
    else:
            optimal_theta = 0.5 * np.arctan(2*B[0][1] / (B[0][0] - B[1][1]))

    cos_theta = np.cos(optimal_theta)
    sin_theta = np.sin(optimal_theta)

    R = np.array([[cos_theta, -sin_theta],
         [sin_theta, cos_theta]])
    
    D = np.round(R.T @ B @ R, 4)
    
    assert D[0][1] == 0 and D[1][0] == 0, "D should be diagonal"

    singular_values = np.sqrt([D[0,0], D[1,1]])
    # Sort singular values in descending order and adjust U, V accordingly
    idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[idx]
    R = R[:, idx]
    U = A @ R @ np.diag(1 / singular_values)
    U = U[:, idx]
    V = R

    return U, singular_values, V.T


a = [[2, 1], [1, 2]]
U, S, V = svd_2x2_singular_values(np.array(a))
print(U, S, V)
# Expected output - (array([[ 0.70710678, -0.70710678], [ 0.70710678, 0.70710678]]), array([3., 1.]), array([[ 0.70710678, 0.70710678], [-0.70710678, 0.70710678]]))
# Current output - (array([[ 0.70710678, -0.70710678], [ 0.70710678, 0.70710678]]), array([3., 1.]), array([[ 0.70710678, 0.70710678], [-0.70710678, 0.70710678]]))

a = [[1, 2], [3, 4]]
U, S, V = svd_2x2_singular_values(np.array(a))
print(U, S, V)
# Expected output - array([[ 0.40455358, 0.9145143 ], [ 0.9145143 , -0.40455358]]), array([5.4649857 , 0.36596619]), array([[ 0.57604844, 0.81741556], [-0.81741556, 0.57604844]])
# Current output - (array([[-0.91462101, 0.40455337], [ 0.40460079, 0.91451382]]), array([0.36592349, 5.46498856]), array([[ 0.81741556, -0.57604844], [ 0.57604844, 0.81741556]]))