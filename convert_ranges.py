import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:


    raveled_values = values.ravel()
    min_x = min(raveled_values)
    max_x = max(raveled_values)

    new_values = []
    for x in raveled_values:
        slope = (x - min_x)/(max_x - min_x)
        new_x = c + slope*(d-c)
        new_values.append(new_x)

    values = np.reshape(new_values, values.shape)
    return values


# x = np.array([0, 5, 10])
# c, d = 2, 4
# print(convert_range(x, c, d))

seq = np.array([[2028, 4522], [1412, 2502], [3414, 3694], [1747, 1233], [1862, 4868]]) 
c, d = 4, 8 
out = convert_range(seq, c, d) 
print(np.round(out, 6))