import numpy as np
import matplotlib.image as mpimg
import os
import json
import algorithms

import convolution_neuron as cn

def load_data(path: str, grayscale: bool = True):
    images = []

    

    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = mpimg.imread(os.path.join(path, filename))

            if grayscale and len(image.shape) == 3:
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

            height, width = image.shape[:2]

            if height != width:
                algorithms.add_padding_to_image(image)
            algorithms.resize_image(image)

            images.append(image)

    return np.array(images)


def save_kernels(nodes: np.ndarray):
    kernels = [ node.kernel.tolist() for node in nodes ]
    with open('src/kernels.json', "w") as f:
        json.dump(kernels, f)


def load_kernels() -> np.ndarray:
    with open('src/kernels.json', "r") as f:
        return np.array(json.load(f))

convolution_nodes = np.array([])

if __name__ == '__main__':
    
    """
    Fetching training data, and then scaling them all down to a set size. Adds padding if needed.
    """
    image_height, image_width = 224
    images = load_data('training_data', grayscale=True, image_height = image_height, image_width = image_width)


    num_convolution_nodes = 2


    """
    Initializing random kernels if no kernels are found, or accessing kernels that are trained/in progress of training
    """
    if os.path.exists('src/kernels.json'):
        kernels = load_kernels()
        convolution_nodes = np.array([cn.ConvolutionNeuron(kernels[i]) for i in range(len(kernels))])
    else:
        file = open('src/kernels.json', 'w')
        file.close()


        for i in range(num_convolution_nodes):
            kernel = np.random.randn(3, 3)
            convolution_nodes: np.ndarray[cn.ConvolutionNeuron] = np.append(convolution_nodes, cn.ConvolutionNeuron(kernel))

        save_kernels(convolution_nodes)


    """
    Training
    """
    for image in images:
        for node in convolution_nodes:
            node.convolve2d(image, nonlinear=True, pooling=True)
            # code that connects the convolved activation matricies to the fully connected layer
            # determine cost using a set function -> cost returns a list matricies of adjustments for each node before the output node
            # backpropagate using recursion, calculating the adjustments neede for the previous connected nodes
            # repeat
