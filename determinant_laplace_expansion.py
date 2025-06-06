def determinant_3x3(matrix: list[list[int | float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def determinant_4x4(matrix: list[list[int | float]]) -> float:
    n_cols = 4
    det = 0

    for i in range(n_cols):
        sub_matrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        s = matrix[0][i] * determinant_3x3(sub_matrix)

        det += ((-1)**i) * s

    return det


a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# a = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
print(determinant_4x4(a))  # Output: 1
