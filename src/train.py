import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import algorithms

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
    # Fetching training data
    images = load_images('training_data', grayscale=True)

    model.load_model()
    debug = True
    
    # Training
    for index, image in enumerate(images):
        # First Convolution Layer (outputs a 3D tensor)
        image_width = np.size(image, 0) 
        image_height = np.size(image, 1)
        sample_cn_one = model.model["cn_layer_one"][0]
        layer_one_output = np.zeros((len(model.model["cn_layer_one"]), 
                                    ((image_width - sample_cn_one.kernel.shape[0]) // sample_cn_one.stride),
                                    ((image_height - sample_cn_one.kernel.shape[1]) // sample_cn_one.stride)
                                    ), dtype=float)
        algorithms.debug_message("Initialized first convolution layer output with shape: " + str(layer_one_output.shape), debug)
        for i, cn in enumerate(model.model["cn_layer_one"]):
            cn.convolve2d(image, nonlinear=True, pooling=True)
            layer_one_output[i] = cn.activation
            algorithms.debug_message(f"Convolved Image {i} out of {len(model.model['cn_layer_one'])}", debug)
        algorithms.debug_message("First layer convolution successful.", debug)

        # Second Convolution Layer (ouputs a 3D tensor)
        sample_cn_two = model.model["cn_layer_two"][0]
        layer_two_output = np.zeros((len(model.model["cn_layer_two"]),
                                    ((layer_one_output.shape[1] - sample_cn_two.kernel.shape[0]) // sample_cn_two.stride), 
                                    ((layer_one_output.shape[2] - sample_cn_two.kernel.shape[1]) // sample_cn_two.stride),                                    
                                    ), dtype=float)
        algorithms.debug_message("Initialized second convolution layer output with shape: " + str(layer_two_output.shape), debug)
        for i, cn in enumerate(model.model["cn_layer_two"]):
            cn.convolve3d(layer_one_output, nonlinear=True, pooling=True)
            layer_two_output[i] = cn.activation
            algorithms.debug_message(f"Convolved Image {i} out of {len(model.model['cn_layer_two'])}", debug)
        algorithms.debug_message("Second layer convolution successful.", debug)

        # Flattening output for fully connected layer.
        flattened_output = layer_two_output.flatten()
        algorithms.debug_message(f"Flattened output to shape: {flattened_output.shape}", debug)


        # code that connects the convolved activation matricies to the fully connected layer
        # determine cost using a set function -> cost returns a list matricies of adjustments for each node before the output node
        # backpropagate using recursion, calculating the adjustments neede for the previous connected nodes
        # repeat

        # ** the model should output: the x, y, width, height of where the face is located.
        # later, the box made will scale back up to the original un-scaled image and the
        # new box will be drawn over the original image.
