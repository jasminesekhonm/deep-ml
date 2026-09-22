import numpy as np 

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape 
    kernel_height, kernel_width = kernel.shape 

    output_height = (input_height + 2*padding - kernel_height ) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    if padding > 0:
        padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode="constant")
    else:
        padded_input = input_matrix

    output = np.zeros((output_height, output_width), dtype=np.float64)

    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride 
            start_j = j * stride 
            output[i, j] = np.sum(padded_input[start_i:start_i+kernel_height, start_j:start_j+kernel_width] * kernel)
    
    return output

