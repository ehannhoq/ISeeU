import os

import numpy as np
import matplotlib.image as mpimg
import cv2 as cv

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
    output = output / 255

    scale_w = new_width / current_width
    scale_h = new_height / current_height
    return output, scale_w , scale_h

def load_wider_data_set(imageset_master_path:str, annotation_file_path:str, target_size:tuple ,batch_size:int, max_faces:int, start_index:int = 0):
    """
    Loads a batch of images and their corresponding bounding box annotations from the WIDER dataset.

    Args:
        imageset_master_path (str): The path to the directory containing the image files.
        annotation_file_path (str): The path to the file containing the annotation data.
        target_size (tuple): The target size (height, width) to which images should be resized.
        batch_size (int): The number of images to load in one batch.
        max_faces (int): The maximum number of faces to consider per image.
        start_index (int, optional): The starting index in the annotation file from which to read. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - images (np.ndarray): A batch of resized images with shape (batch_size, 1, target_height, target_width).
            - expected (list): A list of numpy arrays, each containing bounding box coordinates for faces in the corresponding image.
            - end_index (int): The index in the annotation file where the batch loading stopped, or -1 if the end of the file was reached.
    """

    with open(annotation_file_path, mode='r') as f:
        lines = f.readlines()
    
    images = np.zeros((batch_size, 1, target_size[0], target_size[1]))
    expected = []

    i = 0
    line_index = start_index
    while line_index < len(lines) and i < batch_size:
        image_name = lines[line_index].strip()
        image_path = os.path.join(imageset_master_path, image_name)
        image = mpimg.imread(image_path)

        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        
        image, scale_w, scale_h = resize_image(image=image, target_size=target_size)
        images[i, :] = image
        
        line_index += 1

        num_faces = int(lines[line_index].strip())

        if num_faces > max_faces:
            line_index += 1 + num_faces
            continue

        line_index += 1

        if num_faces == 0:
            line_index += 1

        bounding_boxes = np.zeros((num_faces, 4))

        for j in range(num_faces):
            bbox_info = list(map(int, lines[line_index].split()))
            x, y, w, h = bbox_info[:4]
            x = int(x * scale_w)
            y = int(y * scale_h)
            w = int(w * scale_w)
            h = int(h * scale_h)

            bounding_boxes[j] = np.array([x, y, w, h])
            
            line_index += 1

        expected.append(bounding_boxes)

        i += 1
        print(f"\rImage {i}/{batch_size} loaded", end="")
    

    end_index = line_index if line_index < len(lines) else -1
    print("")

    return images, expected, end_index

def convolve(input: np.ndarray, filter: np.ndarray, stride: int, padding: int) -> np.ndarray:
    batch_size, _, height, width = input.shape
    output_channels, _, filter_height, filter_width = filter.shape

    input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    output_height = (height + 2 * padding - filter_height) // stride + 1
    output_width = (width + 2 * padding - filter_width) // stride + 1

    output = np.zeros((batch_size, output_channels, output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[:, :, y, x] = np.tensordot(
                input[:, :, y * stride:y * stride + filter_height, x * stride:x * stride + filter_width],
                filter,
                axes=([1, 2, 3], [1, 2, 3])
            )

    return output

def leaky_relu(x: np.ndarray, alpha=0.1) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha=0.1) -> np.ndarray:
    return np.where(x > 0, 1, alpha)

def max_pooling(input: np.ndarray, pool_size:tuple) -> np.ndarray:
    batch_size, channels, height, width = input.shape
    output_height = (height - pool_size[0]) // pool_size[0] + 1
    output_width = (width - pool_size[1]) // pool_size[1] + 1

    output = input.reshape(
        (batch_size, channels,
         output_height, pool_size[0],
         output_width, pool_size[1])
    )
    output = np.max(output, axis=(3, 5))
    return output


def sigmoid(input: np.ndarray):
    return 1 / (1 + np.exp(-input))
    

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_int = max(x1, x2)
    y1_int = max(y1, y2)
    x2_int = min(x1 + w1, x2 + w2)
    y2_int = min(y1 + h1, y2 + h2)

    w_int = max(0, x2_int - x1_int)
    h_int = max(0, y2_int - y1_int)

    intersection_area = w_int * h_int

    union_area = (w1 * h1 + w2 * h2) - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

    

def assign_ground_truth(predicted, expected, iou_threshold = 0.9):
    batch_size, predicted_faces, _ = predicted.shape
    output = np.zeros((batch_size, predicted_faces))

    for b in range(batch_size):
        for i, e_box in enumerate(expected[b]):
            iou = calculate_iou(predicted[b, i], e_box)
            if iou > iou_threshold:
                output[b, i] = 1
        
    return output


def binary_cross_entropy_gradient(y_pred, y_true, expected):
    batch_size, predicted_faces = y_pred.shape
    output = np.zeros_like(y_true)

    
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    for b in range(batch_size):
        for i, true_val in enumerate(y_true[b]):
            output[b, i] = -(true_val / y_pred[b, i]) + (1 - true_val) / (1 - y_pred[b, i])
        for j in range(len(expected[b]), predicted_faces):
            output[b, j] = 0
    return output


def mse_gradient(predicted: np.ndarray, expected:list):
    output = np.zeros_like(predicted)
    for b in range(predicted.shape[0]):
        for i, e_box in enumerate(expected[b]):
            output[b, i] = 2 * (predicted[b, i] - e_box)
    return output

            
def kernel_gradient(input: np.ndarray, error: np.ndarray, kernel: tuple):
    in_channels = input.shape[1]
    _, error_channels, out_height, out_width = error.shape
    _, _, kernel_height, kernel_width = kernel


    output = np.zeros((error_channels, in_channels, kernel_height, kernel_width))

    for y in range(out_height):
        for x in range(out_width):
            input_patch = input[:, :, y:y + kernel_height, x:x + kernel_width]
            err_slice = error[:, :, y, x]
            output += np.einsum('bo,bcih->ocih', err_slice, input_patch)

    return output


def convolve_gradient(error: np.ndarray, kernel: np.ndarray, reverse_max_pooling: bool):
    if reverse_max_pooling:
        error = max_unpooling(error, 2)

    kernel = np.flip(kernel, axis=(2, 3))

    batch_size, _, error_height, error_width = error.shape
    kernel_out_channels, _, kernel_height, kernel_width = kernel.shape

    output = np.zeros((batch_size, kernel_out_channels, error_height, error_width))


    for b in range(batch_size):
        for ko in range(kernel_out_channels):
            for y in range(error_height - kernel_height + 1):
                for x in range(error_width - kernel_width + 1):
                    patch = error[b, ko, y:y + kernel_height, x:x + kernel_width]
                    output[b, ko, y, x] = np.sum(patch * kernel[ko, :])
    return output
                    

def max_unpooling(input: np.ndarray, pool_size: int):
    return np.repeat(np.repeat(input, pool_size, axis=2), pool_size, axis=3)