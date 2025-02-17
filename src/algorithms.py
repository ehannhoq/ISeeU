import numpy as np
import cv2 as cv

def convolve2d(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert len(kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape

    output_width = input_width - kernel_width
    output_height = input_height - kernel_height
    activation = np.zeros((output_height, output_width), dtype=float)

    for y in range(output_height):
        for x in range(output_width):
            window = input[y:y + kernel_height, x:x + kernel_width]
            activation[y, x] = np.sum(window * kernel)
    
    return activation


def convolve3d(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert len(kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

    input_depth, input_height, input_width = input.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape

    assert kernel_depth == input_depth, "Kernel and input shape mismatch"

    output_width = input_width - kernel_width
    output_height = input_height - kernel_height

    output = np.zeros((output_height, output_width), dtype=float)


    for y in range(output_height):
        for x in range(output_width):
            cube = input[:, y:y + kernel_height, x:x + kernel_width]
            output[y, x] = np.sum(cube * kernel)

    return output


def convolve_gradient(error_gradient: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    assert len(kernel.shape) == len(error_gradient.shape), "Kernel and error gradient dimension mismatch"

    error_height, error_width = error_gradient.shape
    kernel_height, kernel_width = kernel.shape

    output_width = error_width + kernel_width - 1
    output_height = error_height + kernel_height - 1

    output = np.zeros((output_height, output_width), dtype=float)

    for y in range(output_height):
        for x in range(output_width):
            output[y:y + kernel_height, x:x + kernel_height] += error_gradient[y, x] * kernel

    return output


def leaky_relu(input, alpha: float = 0.1):
    return np.where(input > 0, input, alpha * input)


def max_pooling(input: np.ndarray, size: int = 2, stride: int = 2) -> np.array:
    assert len(input.shape) == 2, "Input must be 2D"

    input_height, input_width = input.shape

    output_width = (input_width - size) // stride + 1
    output_height = (input_height - size) // stride + 1

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            window = input[ y * stride:y * stride + size, x * stride:x * stride + size ]
            output[y, x] = np.max(window)
        
    return output


def mean_squared_error_gradient(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (predicted - expected)


def leaky_relu_gradient(input, alpha: float = 0.1):
    return np.where(input > 0, 1, alpha)    


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    current_height, current_width = image.shape

    if current_height > target_size[0] or current_width < target_size[1]:
    
        if current_width > current_height:
            aspect_ratio = current_width / current_height
            new_width = target_size[1]
            new_height = int(aspect_ratio * new_width)
        else:
            aspect_ratio = current_height / current_width
            new_height = target_size[0]
            new_width = int(aspect_ratio * new_height)
            

    resized_image = cv.resize(src=image, dsize=(new_width, new_height))

    output = np.zeros(target_size, dtype=image.dtype)
    output[ 0:resized_image.shape[0], 0:resized_image.shape[1] ] = resize_image
    return output
