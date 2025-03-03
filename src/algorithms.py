import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 as cv

import model

def convolve(input: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    batch_size, _, input_height, input_width = input.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape

    input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1

    output = np.zeros((batch_size, output_channels, output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[:, :, y, x] = np.tensordot(
                input[:, :, y * stride:y * stride + kernel_height, x * stride:x * stride + kernel_width], kernel, axes=([1, 2, 3], [1, 2, 3])
            )

    return output


def convolve_gradient(error_gradient: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    error_gradient = reverse_max_pooling(error_gradient, pool_size=2)
    batch_size, _, error_gradient_height, error_gradient_width = error_gradient.shape
    _, cn2_channels, kernel_height, kernel_width = kernel.shape
    
    gradient = np.zeros((batch_size, cn2_channels, kernel_height, kernel_width))
    for b in range(batch_size):
        for cn2 in range(cn2_channels):
            for y in range(error_gradient_height - kernel_height + 1):
                for x in range(error_gradient_width - kernel_width + 1):
                    gradient[b] = np.tensordot(
                        error_gradient[b, :, y:y + kernel_height, x:x + kernel_width], kernel[:, cn2], axes=([0, 1, 2], [0, 1, 2])
                    )
                    
    return gradient


def kernel_gradient(input_activation: np.ndarray, error_gradient: np.ndarray, kernel_size: tuple):
    _, input_activation_channels, _, _ = input_activation.shape
    _, error_gradient_channels, error_gradient_height, error_gradient_width = error_gradient.shape
    _, _, kernel_height, kernel_width = kernel_size
    gradient = np.zeros((error_gradient_channels, input_activation_channels, kernel_height, kernel_width))

    for y in range(kernel_height):
        for x in range(kernel_width):
            window = input_activation[:, :, y:y + error_gradient_height, x:x + error_gradient_width]
            gradient[:, :, y, x] = np.sum(window[:, None, :, :, :] * error_gradient[:, :, None, :, :], axis=(0, 3, 4))

    return gradient
    

def leaky_relu(input, alpha: float = 0.1):
    return np.where(input > 0, input, alpha * input)


def leaky_relu_gradient(input, alpha: float = 0.1):
    return np.where(input > 0, 1, alpha)    


def sigmoid(input):
    input = np.clip(input, -500, 500)
    return 1 / (1 + np.exp(-input))


def max_pooling(input: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    batch_size, channels, input_height, input_width = input.shape
    output_height = (input_height - size) // stride + 1
    output_width = (input_width - size) // stride + 1

    output = input.reshape(
        (batch_size, channels,
        output_height, stride, 
        output_width, stride)
    )
    output = np.max(output, axis=(3, 5))
    
    return output

def reverse_max_pooling(input:np.ndarray, pool_size: int = 2):
    return np.repeat(np.repeat(input, pool_size, axis=2), pool_size, axis=3)
    
def mean_squared_error_gradient(predicted: np.ndarray, expected: list) -> np.ndarray:
    batch_size, max_boxes, _ = predicted.shape

    max_faces = max(len(faces) for faces in expected)

    expected_resized = np.zeros((batch_size, max_boxes, 4))

    for batch in range(batch_size):
        num_faces = len(expected[batch])
        expected_resized[batch, :num_faces, :] = np.array(expected[batch])

    return 2 * (predicted - expected_resized)

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

    union_area = (w1 * h1 + w2 * h2) - intersection_area

    return intersection_area / union_area if union_area != 0 else 0


def assign_ground_truth(predicted, expected, iou_threshold = 0.5):
    batch_size, predicted_num_faces, _ = predicted.shape

    output = np.zeros((batch_size, predicted_num_faces))
    tp_list, fp_list, fn_list = [], [], []

    for b in range(batch_size):
        tp, fp, fn = 0, 0, 0
        matched_gt = np.zeros(len(expected[b]))

        for p_box in predicted[b]:
            best_iou = 0
            best_gt_index = -1

            for e_index, e_box in enumerate(expected[b]):
                current_iou = iou(p_box, e_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_index = e_index

            if best_iou > iou_threshold:
                output[b] = 1
                matched_gt[best_gt_index] = 1
                tp += 1
            else:
                fp += 1
        
        fn = np.sum(1 - matched_gt)

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    plot_metrics(tp_list, fp_list, fn_list)

    return output


def plot_metrics(tp_list, fp_list, fn_list):
    plt.ion()
    plt.clf()

    batches = range(1, len(tp_list) + 1)
    
    plt.bar(batches, tp_list, color='green', label='True Positives')
    plt.bar(batches, fp_list, bottom=tp_list, color='red', label='False Positives')
    plt.bar(batches, fn_list, bottom=np.array(tp_list) + np.array(fp_list), color='blue', label='False Negatives')

    plt.xlabel("Batch")
    plt.ylabel("Count")
    plt.title("Detection Metrics per Batch")
    plt.legend()
    
    plt.pause(0.1)
    plt.show(block=False)


def binary_cross_entropy_gradient(ground_truth, predicted):
    epislon = 1e-10
    return -(ground_truth / (predicted + epislon) - (1 - ground_truth) / (1 - predicted + epislon))
   

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


def load_wider_data_set(imageset_master_path:str, annotation_file_path:str, target_size:tuple ,batch_size:int, max_faces:int, start_index:int = 0):
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

        image, aspect_ratio = resize_image(image=image, target_size=target_size)
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
            bounding_boxes[j] = np.array([x, y, w, h])
            bounding_boxes[j] /= aspect_ratio
            line_index += 1

        expected.append(bounding_boxes)

        i += 1
        print(f"\rImage {i}/{batch_size} loaded", end="")
    

    end_index = line_index if line_index < len(lines) else -1
    print("")

    return images, expected, end_index

