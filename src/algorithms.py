import os
import numpy as np
import matplotlib.image as mpimg
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
    kernel_height, kernel_width = kernel.shape
    error_gradient = reverse_max_pooling(error_gradient, pool_size=2)
    error_gradient_height, error_gradient_width = error_gradient.shape

    output_height = error_gradient_height + kernel_height + 1
    output_width = error_gradient_width + kernel_width + 1

    output = np.zeros((output_height, output_width), dtype=float)   

    for y in range(error_gradient_height):
        for x in range(error_gradient_width):
            output[y:y + kernel_height, x:x + kernel_width] += (
                error_gradient[y, x] * kernel
            )

    return output

def reverse_max_pooling(input:np.ndarray, pool_size: int = 2):
    input_height, input_width = input.shape
    output_height = input_height * pool_size
    output_width = input_width * pool_size

    output = np.zeros((output_height, output_width), dtype=float)

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.max(input[y // pool_size, x // pool_size])

    return output



def compute_kernel_gradient(input_activation: np.ndarray, error_gradient: np.ndarray, kernel_size: tuple):
    kernel_height, kernel_width = kernel_size
    gradient = np.zeros((kernel_height, kernel_width))

    for i in range(kernel_height):
        for j in range(kernel_width):
            gradient[i, j] = np.sum(input_activation[i:i+error_gradient.shape[0], j:j+error_gradient.shape[1]] * error_gradient)

    return gradient
    

def leaky_relu(input, alpha: float = 0.1):
    return np.where(input > 0, input, alpha * input)

def sigmoid(input):
    input = np.clip(input, -500, 500)
    return 1 / (1 + np.exp(-input))


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
            current_iou = iou(predicted, actual)
            max_iou = max(current_iou, max_iou)

        if max_iou > iou_threshold:
            confidence_labels.append(1)
        else:
            confidence_labels.append(0)

    return confidence_labels


def binary_cross_entropy_gradient(ground_truth, predicted):
    output = []
    epislon = 1e-10

    for i in range(len(ground_truth)):
        output.append( np.where(ground_truth[i] == 1, -1 / predicted[i] + epislon, 1 / (1 - predicted[i] + epislon)) )

    return np.array(output)


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    current_height, current_width = image.shape
    target_height, target_width = target_size

    new_height = target_height
    new_height = target_width
    
    if current_height > target_height or current_width > target_width:

        if current_width > current_height:
            new_width = target_width
            aspect_ratio = current_height / current_width
            new_height = int(aspect_ratio * new_width)
        else:
            aspect_ratio = current_width / current_height
            new_height = target_height
            new_width = int(aspect_ratio * new_height)
            
    if new_height == 0 or new_width == 0:
        raise Exception("Calculated new height or width is 0.")

    resized_image = cv.resize(src=image, dsize=(new_width, new_height))

    output = np.zeros(target_size, dtype=image.dtype)
    output[ 0:resized_image.shape[0], 0:resized_image.shape[1] ] = resized_image
    return (output, new_width / new_height)


def load_wider_data_set(imageset_master_path:str, annotation_file_path:str, batch_size:int, start_index:int = 0):
    dataset = []

    with open(annotation_file_path, mode='r') as f:
        lines = f.readlines()
    
    i = start_index
    images_loaded = 0
    while i < len(lines) and images_loaded < batch_size:

        image_name = lines[i].strip()
        image_path = os.path.join(imageset_master_path, image_name)
        original_image = mpimg.imread(image_path)
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        if num_faces == 0:
            i += 1

        bounding_boxes = []
        for _ in range(num_faces):
            bbox_info = list(map(int, lines[i].split()))
            x, y, w, h = bbox_info[:4]
            bounding_boxes.append( (x, y, w, h) )
            i += 1

        dataset.append( {
            "image": original_image,
            "bounding_boxes": np.array(bounding_boxes)
            } )

        images_loaded += 1
        print(f"\rLoaded image {i} | {images_loaded}/{batch_size}", end="")
    

    end_index = i if i != len(lines) - 1 else -1
    print("")
    return dataset, end_index