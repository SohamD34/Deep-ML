import numpy as np
from typing import Tuple

def gini_impurity(y: np.ndarray) -> float:
    """Calculate the Gini impurity for a set of labels."""
    if len(y) == 0:
        return 0.0
    
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    gini = 1 - np.sum(probabilities ** 2)
    return gini


def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    
    num_features = X.shape[1]
    best_gini = float('inf')
    best_split = (-1, -1.0)

    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            
            if np.any(left_indices) and np.any(right_indices):
                gini_left = gini_impurity(y[left_indices])
                gini_right = gini_impurity(y[right_indices])
                
                weighted_gini = (len(y[left_indices]) * gini_left + 
                                 len(y[right_indices]) * gini_right) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (feature_index, threshold)
    
    return best_split


X = np.array([[2.5],[3.5],[1.0],[4.0]])
y = np.array([0,1,0,1])
print(find_best_split(X, y))