# Implement Early Stopping Based on Validation Loss
# Create a function to decide when to stop training a model early based on a list of validation losses. 
# The early stopping criterion should stop training if the validation loss hasn't improved for a specified number of epochs (patience), and only count as improvement if the loss decreases by more than a certain threshold (min_delta). 
# Your function should return the epoch to stop at and the best epoch that achieved the lowest validation loss.

# Example:
# Input:
# [0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78], patience=2, min_delta=0.01

# Output:
# (4, 2)

# Reasoning:
# The best validation loss is 0.75 at epoch 2. There is no improvement greater than 0.01 for the next 2 epochs. Therefore, training should stop at epoch 4.

from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine the epoch to stop training based on validation losses.

    Parameters:
    val_losses (list[float]): List of validation losses for each epoch.
    patience (int): Number of epochs to wait for improvement before stopping.
    min_delta (float): Minimum change in validation loss to qualify as an improvement.

    Returns:
    Tuple[int, int]: The epoch to stop at and the best epoch with the lowest validation loss.
    """
    best_epoch = 0
    best_loss = float('inf')
    last_improvement_epoch = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            last_improvement_epoch = epoch

        if epoch - last_improvement_epoch >= patience:
            return epoch, best_epoch

    return len(val_losses) - 1, best_epoch  


print(early_stopping([0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78], 2, 0.01))
# Expected - (4, 2)

print(early_stopping([0.9, 0.8, 0.79, 0.78, 0.77], 2, 0.1))
# Expected - (4, 2)

print(early_stopping([0.9, 0.8, 0.7, 0.6, 0.5], 2, 0.01))
# Expected -  (4, 4)

print(early_stopping([0.5, 0.4], 3, 0.01))
# Expected - (1, 1)

print(early_stopping([0.5, 0.4, 0.4, 0.4, 0.4], 2, 0.01))
# Expected - (3, 1)