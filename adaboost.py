import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    for _ in range(n_clf):
        min_error = float('inf')
        best_clf = {}

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[X[:, feature] < threshold] = -1
                    predictions *= polarity

                    error = np.sum(w * (predictions != y))

                    if error < min_error:
                        min_error = error
                        best_clf = {
                            'polarity': polarity,
                            'threshold': threshold,
                            'feature_index': feature
                        }

        # Compute alpha
        alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))
        best_clf['alpha'] = alpha

        clfs.append(best_clf)

        # Update weights
        predictions = np.ones(n_samples)
        predictions[X[:, best_clf['feature_index']] < best_clf['threshold']] = -1
        predictions *= best_clf['polarity']

        w *= np.exp(-alpha * y * predictions)
        w /= np.sum(w)

    return clfs


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])
n_clf = 3
clfs = adaboost_fit(X, y, n_clf)
print(clfs)

# Expected output:
# [{'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}, {'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}, {'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}]

X = np.array([[8, 7], [3, 4], [5, 9], [4, 0], [1, 0], [0, 7], [3, 8], [4, 2], [6, 8], [0, 2]])
y = np.array([1, -1, 1, -1, 1, -1, -1, -1, 1, 1])
n_clf = 2
clfs = adaboost_fit(X, y, n_clf)
print(clfs)

# Expected output:
# [{'polarity': 1, 'threshold': 5, 'feature_index': 0, 'alpha': 0.6931471803099453}, {'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 0.5493061439673882}]