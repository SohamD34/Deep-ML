import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    
    h_o, w_o = input_matrix.shape
    h_k, w_k = kernel.shape

    h_i = (h_o + 2*padding - h_k) // stride + 1
    w_i = (w_o + 2*padding - w_k) // stride + 1

    output_matrix = np.array([[0 for i in range(w_i)] for j in range(h_i)])

    for i in range(h_i):
        for j in range(w_i):

            part_of_image = np.array([[input_matrix[y][x] for x in range(j, j+w_k)] for y in range(i, i+h_k)])
            
            # print(part_of_image)
            # print(kernel)

            dot_prod = 0

            for y in range(h_k):
                for x in range(w_k):
                    dot_prod += part_of_image[y][x] * kernel[y][x]

            # print(dot_prod)

            output_matrix[i][j] = dot_prod
    
    return output_matrix


input_matrix = np.array([ [1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], ]) 
kernel = np.array([ [1., 2.], [3., -1.], ]) 
padding, stride = 0, 1 
output = simple_conv2d(input_matrix, kernel, padding, stride) 
print(output)
