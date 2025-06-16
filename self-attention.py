import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    
    return Q, K, V



def self_attention(Q, K, V):
	
    Q = Q / np.sqrt(Q.shape[-1])
    scores = np.matmul(Q, K.T)
    scores = np.exp(scores)
    scores = scores / np.sum(scores, axis=-1, keepdims=True)
    output = np.matmul(scores, V)

    return output



X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)