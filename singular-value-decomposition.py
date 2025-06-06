import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:

    ATA = A.T @ A
    AAT = A @ A.T

    eigenvals_AAT, eigvectors_AAT = np.linalg.eigh(AAT)
    eigenvals_ATA, eigvectors_ATA = np.linalg.eigh(ATA)

    Sigma = np.sqrt(np.abs(eigenvals_AAT))
    Sigma = np.sort(Sigma)[::-1]  

    U = eigvectors_AAT
    V_T = eigvectors_ATA.T

    return U, Sigma, V_T


a = np.array([[2, 1], [1, 2]])

U, Sigma, V_T = svd_2x2_singular_values(a)
print("U:\n", U)
print("Sigma:\n", Sigma)
print("V_T:\n", V_T)


U, Sigma, V_T = np.linalg.svd(a)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V_T:\n", V_T)