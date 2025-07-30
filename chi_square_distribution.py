import numpy
import math


def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x < 0 or k <= 0:
            return 0.0
    coeff = 1 / (math.pow(2, k / 2) * math.gamma(k / 2))
    probability = coeff * math.pow(x, (k / 2) - 1) * math.exp(-x / 2)
    return round(probability, 3)


# Tests for chi_square_probability
# test 1
x = 2
k = 2
print(chi_square_probability(x, k))  # Expected output: 0.184

# test 2
x = 0
k = 4
print(chi_square_probability(x, k))  # Expected output: 0.0

# test 3
x = 5
k = 3
print(chi_square_probability(x, k))  # Expected output: 0.104

# test 4
x = 10
k = 6
print(chi_square_probability(x, k))  # Expected output: 0.050