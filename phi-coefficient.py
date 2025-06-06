'''
If two arrays are binary
x = []
y = []

Then we compute the count matrix as follows:
|         | y=0 | y=1 |
|---------|-----|-----|
| x=0     | a   | b   |
| x=1     | c   | d   |


The phi coefficient is a measure of association for two binary variables. It is defined as:
phi = (ad - bc) / sqrt((a + b)(c + d)(a + c)(b + d))
'''

def phi_corr(x: list[int], y: list[int]) -> float:

    a = sum(1 for i in range(len(x)) if x[i] == 0 and y[i] == 0)
    b = sum(1 for i in range(len(x)) if x[i] == 0 and y[i] == 1)
    c = sum(1 for i in range(len(x)) if x[i] == 1 and y[i] == 0)
    d = sum(1 for i in range(len(x)) if x[i] == 1 and y[i] == 1)

    numerator = (a * d) - (b * c)
    denominator = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5    
    if denominator == 0:
        return 0.0  # Avoid division by zero, return 0 if no association

    val = numerator / denominator
    
    return round(val,4)