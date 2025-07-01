import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    
    dot_prod = np.sum(true_labels * np.log(predicted_probs + epsilon))
    loss = -dot_prod / true_labels.shape[0]
    return loss
    

predicted_probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
true_labels = [[1, 0, 0], [0, 1, 0]]
print(compute_cross_entropy_loss(np.array(predicted_probs), np.array(true_labels)))
# Ans- 0.4338
