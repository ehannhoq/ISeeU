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
    
    # Training
    for index, image in enumerate(images):


        ## TODO: Convolution layer output sizes does not account for strides and pooling correctly.

        # First Convolution Layer (outputs a 3D tensor)
        image_width = np.size(image, 0) 
        image_height = np.size(image, 1)
        sample_convolution_neuron_l1 = model.model["cn_layer_one"][0]
        cn_l1_output = np.zeros((len(model.model["cn_layer_one"]), 
                                    ((image_width - sample_convolution_neuron_l1.kernel.shape[0]) // sample_convolution_neuron_l1.stride),
                                    ((image_height - sample_convolution_neuron_l1.kernel.shape[1]) // sample_convolution_neuron_l1.stride)
                                    ), dtype=float)
        for i, convolution_neuron_l1 in enumerate(model.model["cn_layer_one"]):
            convolution_neuron_l1.convolve2d(image, nonlinear=True, pooling=True)
            cn_l1_output[i] = convolution_neuron_l1.activation

        # Second Convolution Layer (ouputs a 3D tensor)
        sample_convolution_neuron_l2 = model.model["cn_layer_two"][0]
        cn_l2_output = np.zeros((len(model.model["cn_layer_two"]),
                                    ((cn_l1_output.shape[1] - sample_convolution_neuron_l2.kernel.shape[0]) // sample_convolution_neuron_l2.stride), 
                                    ((cn_l1_output.shape[2] - sample_convolution_neuron_l2.kernel.shape[1]) // sample_convolution_neuron_l2.stride),                                    
                                    ), dtype=float)
        for i, convolution_neuron_l2 in enumerate(model.model["cn_layer_two"]):
            convolution_neuron_l2.convolve3d(cn_l1_output, nonlinear=True, pooling=True)
            cn_l2_output[i] = convolution_neuron_l2.activation

        # Enforce adaptive pooling to ensure all outputs are the same size, regardless of image input.
        sample_fc_neuron_l1 = model.model["fc_layer_one"][0]
        cn_l2_output = algorithms.adaptive_pooling(cn_l2_output, target_size=(sample_fc_neuron_l1.weights.shape[0], sample_fc_neuron_l1.weights.shape[0]))

        # Flattening adaptively pooled output for fully connected layer.
        flattened_output = cn_l2_output.flatten()

        # First Fully Connected Layer (outputs a 1D array)
        fc_l1_output = np.zeros((len(model.model["fc_layer_one"])))
        for i, fc_neuron_l1 in enumerate(model.model["fc_layer_one"]):
            fc_neuron_l1.calculate_activation(flattened_output)
            fc_l1_output[i] = fc_neuron_l1.activation

        # Second Fully Connected Layer (outputs a 1D array)
        fc_l2_output = np.zeros((len(model.model["fc_layer_two"])))
        for i, fc_neuron_l2 in enumerate(model.model["fc_layer_two"]):
            fc_neuron_l2.calculate_activation(fc_l1_output)
            fc_l2_output[i] = fc_neuron_l2.activation
        
        # Output Layer
        output = np.zeros((len(model.model["output_neurons"])))
        for i, output_neuron in enumerate(model.model["output_neurons"]):
            output_neuron.calculate_activation(fc_l2_output)
            output[i] = output_neuron.activation


        x_neuron = model.model["output_neurons"][0]
        y_neuron = model.model["output_neurons"][1]
        width_neuron = model.model["output_neurons"][2]
        height_neuron = model.model["output_neurons"][3]

        # code that draws a box depending on the x, y, width, and height.
        # backpropagation code
