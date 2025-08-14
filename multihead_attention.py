import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    score_max = np.max(scores, axis=1, keepdims=True) 
    weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis=1, keepdims=True)
    output = np.matmul(weights, V)
    return output


def multi_head_attention(Q, K, V, n_heads):

    if Q.shape[-1] % n_heads == 0:
        d_k = Q.shape[-1] // n_heads
    else:
        raise ValueError("The last dimension of Q must be divisible by n_heads.")
    
    # Reshape Q, K, V for multi-head attention
    Q = np.reshape(Q, (Q.shape[0], n_heads, d_k))   
    K = np.reshape(K, (K.shape[0], n_heads, d_k))
    V = np.reshape(V, (V.shape[0], n_heads, d_k))
    
    outputs = []
    
    for i in range(n_heads):
            output_i = self_attention(Q[i], K[i], V[i])
            outputs.append(output_i)
    
    return np.concatenate(outputs, axis=-1)


Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 0], [0, 1]])
n_heads = 2
print(multi_head_attention(Q, K, V, n_heads))