import numpy as np
import cv2 as cv


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

def resize_image(input: np.ndarray) -> np.ndarray:
    current_width = np.size(input, 0)
    current_height = np.size(input, 1)

    if current_width > current_height:
        return cv.resize(input, (300, int(300 * current_height / current_width)))
    else:
        return cv.resize(input, (int(300 * current_width / current_height), 300))
    
def debug_message(message: str, print: bool) -> None:
    if print:
        print(message)