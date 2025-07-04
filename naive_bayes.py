import numpy as np

# Write a Python class to implement the Bernoulli Naive Bayes classifier for binary (0/1) feature data. 
# Your class should have two methods: forward(self, X, y) to train on the input data (X: 2D NumPy array of binary features, y: 1D NumPy array of class labels) 
# and predict(self, X) to output predicted labels for a 2D test matrix X. 
# Use Laplace smoothing (parameter: smoothing=1.0). Return predictions as a NumPy array. Only use NumPy. 
# Predictions must be binary (0 or 1) and you must handle cases where the training data contains only one class. 
# All log/likelihood calculations should use log probabilities for numerical stability.

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_priors = None
        self.feature_probs = None

    def forward(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = counts / len(y)

        self.feature_probs = {}
        for c in self.classes:
            X_c = X[y == c]
            self.feature_probs[c] = (X_c.sum(axis=0) + self.smoothing) / (X_c.shape[0] + 2 * self.smoothing)

    def predict(self, X):
        feature_prob_matrix = np.array([self.feature_probs[c] for c in self.classes])
        log_class_priors = np.log(self.class_priors)
        log_feature_probs = np.log(feature_prob_matrix)
        log_feature_inv_probs = np.log(1 - feature_prob_matrix)
        log_probs = log_class_priors + X @ log_feature_probs.T + (1 - X) @ log_feature_inv_probs.T
        return self.classes[np.argmax(log_probs, axis=1)]


model = NaiveBayes(smoothing=1.0)
X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]])
y = np.array([1, 1, 0, 0, 1])
model.forward(X, y)
print(model.predict(np.array([[1, 0, 1]])))