import numpy as np 


def svd_2x2_singular_values(A: np.ndarray) -> tuple:

   At = np.transpose(A)
   AtA = At @ A

   V = np.eye(2)

   for _ in range(1):
       # Compute rotation angle for a 2x2 matrix
       if AtA[0,0] == AtA[1,1]:
           theta = np.pi/4
       else:
           theta = 0.5 * np.arctan2(2 * AtA[0,1], AtA[0,0] - AtA[1,1])
       
       # Create rotation matrix
       r = np.array(
           [
               [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]
               ]
           )
       
       # apply rotation
       d = np.transpose(r) @ AtA @ r
       # update AtA
       AtA = d
       # accumulate v
       V = V @ r

   # sigma is the diagonal elements squared
   S = np.sqrt([d[0,0], d[1,1]])
   S_inv = np.array([[1/S[0], 0], [0, 1/S[1]]])
   
   U = A @ V @ S_inv
   
   return (U, S, V.T)


a = np.array([[2, 1], [1, 2]])

U, Sigma, V_T = svd_2x2_singular_values(a)
print("U:\n", U)
print("Sigma:\n", Sigma)
print("V_T:\n", V_T)


U, Sigma, V_T = np.linalg.svd(a)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V_T:\n", V_T)