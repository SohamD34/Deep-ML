import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

    h_t = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for x_t in input_sequence:
        x_t = np.array(x_t)
        a_t = np.dot(Wh, h_t) + np.dot(Wx, x_t) + b
        h_t = np.tanh(a_t)

    return np.round(h_t, 4).tolist()



print(rnn_forward( [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0.0, 0.0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0]], [0.1, 0.2] ))
# Expected - [0.7474, 0.9302]

print(rnn_forward([[0.5], [0.1], [-0.2]], [0.0], [[1.0]], [[0.5]], [0.1]))
# Expected - [0.118]


print(rnn_forward([[1.0], [2.0], [3.0]], [0.0], [[0.5]], [[0.8]], [0.0]))
# Expected - [0.9759]