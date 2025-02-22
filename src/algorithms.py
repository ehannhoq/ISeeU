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

def sigmoid(input):
    return 1 / 1 + np.pow(np.e, -input)


def max_pooling(input: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    input_depth, input_height, input_width = input.shape

    output_height = (input_height - size) // stride + 1
    output_width = (input_width - size) // stride + 1

    output = np.zeros((input_depth, output_height, output_width), dtype=float)

    for d in range(input_depth):
        for y in range(output_height):
            for x in range(output_width):
                window = input[ d, y * stride:y * stride + size, x * stride:x * stride + size ]
                output[d, y, x] = np.max(window)
    
    return output

    
def mean_squared_error_gradient(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return 2 * (predicted - expected)


def leaky_relu_gradient(input, alpha: float = 0.1):
    return np.where(input > 0, 1, alpha)    

def sigmoid_gradient(input):
    pass

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_int = max(x1, x2)
    y1_int = max(y1, y2)
    x2_int = min(x1 + w1, x2 + w2)
    y2_int = min(y1 + h1, y2 + h2)

    w_int = max(0, x2_int - x1_int)
    h_int = max(0, y2_int - y1_int)

    intersection_area = w_int * h_int

    union_area = w1 * h1 + w2 * h2

    return intersection_area / union_area if union_area != 0 else 0


def assign_ground_truth(predicted_boxes, actual_boxes, iou_threshold = 0.5):
    confidence_labels = []

    for predicted in predicted_boxes:
        max_iou = 0
        for actual in actual_boxes:
            iou = iou(predicted, actual)
            max_iou = max(iou, max_iou)

        if max_iou > iou_threshold:
            confidence_labels.append(1)
        else:
            confidence_labels.append(0)

    return confidence_labels


def binary_cross_entropy_gradient(ground_truth, predicted):
    return ( -ground_truth / predicted ) + ( (1 - ground_truth) / (1 - predicted) )


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    current_height, current_width = image.shape
    target_height, target_width = target_size
    
    if current_height > target_height or current_width < target_width:
    
        if current_width > current_height:
            new_width = target_width
            aspect_ratio = current_height / current_width
            new_height = int(aspect_ratio * new_width)
        else:
            aspect_ratio = current_width / current_height
            new_height = target_height
            new_width = int(aspect_ratio * new_height)
            

    resized_image = cv.resize(src=image, dsize=(new_width, new_height))

    output = np.zeros(target_size, dtype=image.dtype)
    output[ 0:resized_image.shape[0], 0:resized_image.shape[1] ] = resized_image
    return output
