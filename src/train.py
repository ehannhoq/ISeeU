import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import json
import algorithms

import convolution_neuron
import neuron
import model

def load_images(path: str, grayscale: bool = True) -> np.ndarray:
    images = []

    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = mpimg.imread(os.path.join(path, filename))

            if grayscale and len(image.shape) == 3:
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

            # For training and testing, we're going to downscale every image from A x B -> C x D, with either C or D
            # being 300 pixels, depending on whether A or B is greater. (ie. 600 x 400 -> 300 x 200, or 400 x 600 -> 200 x 300)
            image = algorithms.resize_image(image)

            images.append(image)

    return np.array(images)

if __name__ == '__main__':
    # Fetching training data, and then scaling them all down to a set size. Adds padding if needed.
    images = load_images('training_data', grayscale=True)

    model.load_model()

    # Training
    for image in images:
        
        for cn in model.model["cn_layer_one"]:
            cn.convolve2d(image, nonlinear=True, pooling=True)
            # code that connects the convolved activation matricies to the fully connected layer
            # determine cost using a set function -> cost returns a list matricies of adjustments for each node before the output node
            # backpropagate using recursion, calculating the adjustments neede for the previous connected nodes
            # repeat

            # ** the model should output: the x, y, width, height of where the face is located.
            # later, the box made will scale back up to the original un-scaled image and the
            # new box will be drawn over the original image.
