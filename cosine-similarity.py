import numpy as np

def cosine_similarity(v1, v2):
	
    cosine = np.dot(v1, v2)
    normalization = np.linalg.norm(v1) * np.linalg.norm(v2) 

    if normalization == 0:
        return 0.0

    return cosine / normalization


v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))