import numpy as np
import matplotlib.image as mpimg
import os
import json

import src.convolution_neuron as cn

def load_data(path: str, grayscale: bool = True):
    images = []

    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = mpimg.imread(os.path.join(path, filename))
            if grayscale and len(image.shape) == 3:
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
          
            images.append(image)

    return np.array(images)

def save_kernels(nodes: np.ndarray):
    kernels = [ node.kernel.tolist() for node in nodes ]
    with open('kernels.json', "w") as f:
        json.dump(kernels, f)

def load_kernels() -> np.ndarray:
    with open('kernels.json', "r") as f:
        return np.array(json.load(f))
    
def leaky_relu(input: np.array, alpha: float = 0.1) -> np.array:
    return np.where(input > 0, input, alpha * input)

def max_pooling(input: np.array, size: int = 2, stride: int = 2) -> np.array:

    output_rows: int = (len(input) - size) // stride + 1
    output_cols: int = (len(input[0]) - size) // stride + 1

    output = np.zeros((output_rows, output_cols))

    for x in range(output_rows):
        for y in range(output_cols):
            window = input[x * stride:x * stride + size, y * stride:y * stride + size]
            output[x, y] = np.max(window)
        
    return output
    
convolution_nodes = np.array([])

if __name__ == '__main__':
    # images = load_data('training_data', grayscale=False)

    if os.path.exists('kernels.json'):
        kernels = load_kernels()
        convolution_nodes = np.array([cn.ConvolutionNeuron(kernels[i]) for i in range(len(kernels))])
    else:
        file = open('kernels.json', 'w')
        file.close()

        num_convolution_nodes = 2

        for i in range(num_convolution_nodes):
            kernel = np.random.randn(3, 3)
            convolution_nodes = np.append(convolution_nodes, cn.ConvolutionNeuron(kernel))

        save_kernels(convolution_nodes)

    current_kernels = [ node.kernel.tolist() for node in convolution_nodes ]
    print(current_kernels)
