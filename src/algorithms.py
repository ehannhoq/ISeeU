import numpy as np


def leaky_relu(input: np.ndarray, alpha: float = 0.1) -> np.array:
    return np.where(input > 0, input, alpha * input)


def max_pooling(input: np.ndarray, size: int = 2, stride: int = 2) -> np.array:

    output_rows: int = (len(input) - size) // stride + 1
    output_cols: int = (len(input[0]) - size) // stride + 1

    output = np.zeros((output_rows, output_cols))

    for x in range(output_rows):
        for y in range(output_cols):
            window = input[x * stride:x * stride + size, y * stride:y * stride + size]
            output[x, y] = np.max(window)
        
    return output

def add_padding_to_image(input: np.ndarray):
    height, width = input.shape[:2]

    size = np.max(height, width)
    padded_image = np.zeros((size, size), dtype=float)

    x_offset = (size - height) // 2
    y_offset = (size - width) // 2

    padded_image[x_offset:x_offset+height, y_offset:y_offset+width] = image
    image = padded_image


def resize_image(input: np.ndarray, target_height: int, target_width: int):
    return
    