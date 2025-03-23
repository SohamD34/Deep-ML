import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	
    means = np.mean(data, axis=0)
    std_devs = data.std(axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)

    standardized_data = np.round((data - means) / std_devs, 4)
    normalized_data = np.round((data - mins) / (maxs - mins), 4)

    return standardized_data, normalized_data