import numpy as np
import cv2 as cv


def leaky_relu(input: np.ndarray, alpha: float = 0.1) -> np.array:
    return np.where(input > 0, input, alpha * input)


def max_pooling(input: np.ndarray, size: int = 2, stride: int = 2) -> np.array:

    assert len(input.shape) == 2, "Input must be 2D"

    input_height, input_width = input.shape

    output_width: int = (input_width - size) // stride + 1
    output_height: int = (input_height - size) // stride + 1

    output = np.zeros((output_height, output_width))

    for x in range(output_width):
        for y in range(output_height):
            window = input[ y * stride:y * stride + size, x * stride:x * stride + size ]
            output[y, x] = np.max(window)
        
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