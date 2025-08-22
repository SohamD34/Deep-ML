import numpy as np

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):

    seq_len = Q.shape[0]
    d_k = Q.shape[1]

    if scale_factor is None:
        scale_factor = np.sqrt(d_k).astype(float)

    output = np.zeros((seq_len, V.shape[1]), dtype=V.dtype)

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)

        local_Q = Q[i:i+1]
        local_K = K[start:end]
        local_V = V[start:end]

        scores = np.dot(local_Q, local_K.T) / scale_factor
        max_score = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_score)

        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        output[i] = np.dot(attention_weights, local_V)

    return output


Q = np.array([[1.0], [1.0], [1.0]])
K = np.array([[1.0], [1.0], [1.0]])
V = np.array([[1.0], [2.0], [3.0]])
print(sparse_window_attention(Q, K, V, 1))
# Expected - [[1.5] [2. ] [2.5]]