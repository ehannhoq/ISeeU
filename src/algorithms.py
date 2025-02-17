import numpy as np
import cv2 as cv

<<<<<<< HEAD

def convolve(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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


def gradient_convolve(error_gradient: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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

=======
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

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

def adaptive_pooling(input: np.ndarray, target_size: tuple) -> np.ndarray:
    assert len(input.shape) == 2, "Input must be 2D"
    
    input_height, input_width = input.shape
    output_height, output_width = target_size

    stride_width = input_width // output_width
    stride_height = input_height // output_height
    size_width = stride_width
    size_height = stride_height
    
    output = np.zeros(target_size, dtype=float)

    for y in range(output_height):
        for x in range(output_width):
            window = input[y * stride_height:y * stride_height + size_height, x * stride_width:x * stride_width + size_width]
            output[y, x] = np.max(window)
    
    return output

def mean_squared_error_gradient(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (predicted - expected)

def leaky_relu_gradient(input, alpha: float = 0.1):
    return np.where(input > 0, 1, alpha)    

<<<<<<< HEAD

def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    current_height, current_width, current_channels = image.shape
=======
def resize_image(input: np.ndarray) -> np.ndarray:
    current_width = np.size(input, 0)
    current_height = np.size(input, 1)
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

    if current_width > current_height:
        return cv.resize(input, (300, int(300 * current_height / current_width)))
    else:
<<<<<<< HEAD
        aspect_ratio = current_height / current_width
        new_height = target_size[0]
        new_width = int(aspect_ratio * new_height)

    resized_image = cv.resize(src=image, dsize=(new_width, new_height))

    output = np.zeros(target_size, dtype=image.dtype)
    output[ 0:resized_image.shape[0], 0:resized_image.shape[1] , :] = resized_image
    return output
=======
        return cv.resize(input, (int(300 * current_width / current_height), 300))
>>>>>>> parent of bbc1e17 (switched from oop to matricies)
