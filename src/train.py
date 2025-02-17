import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import algorithms

import model

def load_images(path: str, grayscale: bool = True) -> np.array:
    image_data = []

    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = mpimg.imread(os.path.join(path, filename))

            if grayscale and len(image.shape) == 3:
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

            # For training and testing, we're going to downscale every image from A x B -> C x D, with either C or D
            # being 300 pixels, depending on whether A or B is greater. (ie. 600 x 400 -> 300 x 200, or 400 x 600 -> 200 x 300)
            image = algorithms.resize_image(image)


            image_data.append(np.array([string_to_numbers(filename.split(".")[0])]), image)

    return np.array(image_data)

def string_to_numbers(string: str) -> list:
    return [int(x) for x in string.split(",")]

if __name__ == '__main__':
    # Fetching training data
    images = load_images('training_data', grayscale=True)

    model.load_model()
    
    # Training
    for data in images:

        # Preprosessing
        expected = data[0]
        image = data[1]
        learning_rate = 0.005

        # First Convolution Layer (outputs a 3D tensor)
        image_width = np.size(image, 0) 
        image_height = np.size(image, 1)
        sample_convolution_neuron_l1 = model.model["cn_layer_one"][0]
        cn_l1_output = np.zeros((len(model.model["cn_layer_one"]), 
                                    ((image_width - sample_convolution_neuron_l1.kernel.shape[0]) // sample_convolution_neuron_l1.stride),
                                    ((image_height - sample_convolution_neuron_l1.kernel.shape[1]) // sample_convolution_neuron_l1.stride)
                                    ), 
                                    dtype=float
                                )
        for i, convolution_neuron_l1 in enumerate(model.model["cn_layer_one"]):
            convolution_neuron_l1.convolve2d(image)
            cn_l1_output[i] = convolution_neuron_l1.activation

        # Applying non-linearity and pooling.
        cn_l1_output = algorithms.leaky_relu(cn_l1_output)
        cn_l1_output = algorithms.max_pooling(cn_l1_output)

        # Second Convolution Layer (ouputs a 3D tensor)
        sample_convolution_neuron_l2 = model.model["cn_layer_two"][0]
        cn_l2_output = np.zeros((len(model.model["cn_layer_two"]),
                                    ((cn_l1_output.shape[1] - sample_convolution_neuron_l2.kernel.shape[0]) // sample_convolution_neuron_l2.stride), 
                                    ((cn_l1_output.shape[2] - sample_convolution_neuron_l2.kernel.shape[1]) // sample_convolution_neuron_l2.stride),                                    
                                    ), 
                                    dtype=float
                                )
        for i, convolution_neuron_l2 in enumerate(model.model["cn_layer_two"]):
            convolution_neuron_l2.convolve3d(cn_l1_output)
            cn_l2_output[i] = convolution_neuron_l2.activation

        # Applying non-linearity and pooling.
        cn_l2_output = algorithms.leaky_relu(cn_l2_output)
        cn_l2_output = algorithms.max_pooling(cn_l2_output)

        # Enforce adaptive pooling to ensure all outputs are the same size, regardless of image input.
        sample_fc_neuron_l1 = model.model["fc_layer_one"][0]
        target_size = (sample_fc_neuron_l1.weights.shape[0], sample_fc_neuron_l1.weights.shape[0])
        cn_l2_output = algorithms.adaptive_pooling(cn_l2_output, target_size=target_size)

        # Flattening adaptively pooled output for fully connected layer.
        flattened_output = cn_l2_output.flatten()

        # First Fully Connected Layer (outputs a 1D array)
        fc_l1_output = np.zeros((len(model.model["fc_layer_one"])), dtype=float)
        for i, fc_neuron_l1 in enumerate(model.model["fc_layer_one"]):
            fc_neuron_l1.calculate_activation(flattened_output)
            fc_l1_output[i] = fc_neuron_l1.activation

        # Applying non-linearity
        fc_l1_output = algorithms.leaky_relu(fc_l1_output)

        # Second Fully Connected Layer (outputs a 1D array)
        fc_l2_output = np.zeros((len(model.model["fc_layer_two"])),dtype=float)
        for i, fc_neuron_l2 in enumerate(model.model["fc_layer_two"]):
            fc_neuron_l2.calculate_activation(fc_l1_output)
            fc_l2_output[i] = fc_neuron_l2.activation

        # Applying non-linearity
        fc_l2_output = algorithms.leaky_relu(fc_l2_output)
        
        # Output Layer
        output = np.zeros((len(model.model["output_layer"])), dtype=float)
        for i, output_neuron in enumerate(model.model["output_layer"]):
            output_neuron.calculate_activation(fc_l2_output)
            output[i] = output_neuron.activation

        predicted_x = output[0].activation
        predicted_y = output[1].activation
        predicted_width = output[2].activation
        predicted_height = output[3].activation

        # Calculating error gradient for each layer.
        output_error_gradient = algorithms.mean_squared_error_gradient(
            np.array(predicted_x, predicted_y, predicted_width, predicted_height), 
            expected
        )

        fc2_error_gradient = np.zeros((len(model.model["fc_layer_two"])), dtype=float)
        for i, fc_l2_neuron in enumerate(model.model["fc_layer_two"]):
            activation_derivative = algorithms.leaky_relu_gradient(fc_l2_neuron.activation)
            fc2_error_gradient[i] = np.dot(output_error_gradient - fc_l2_neuron.weights) * activation_derivative

        
        fc1_error_gradient = np.zeros((len(model.model["fc_layer_one"])), dtype=float)
        for i, fc_l1_neuron in enumerate(model.model["fc_layer_one"]):
            activation_derivative = algorithms.leaky_relu_gradient(fc_l1_neuron.activation)
            fc1_error_gradient[i] = np.dot(fc2_error_gradient - fc_l1_neuron.weights) * activation_derivative


        fc1_error_gradient_3d = fc1_error_gradient.reshape(cn_l2_output.shape)
        cn2_error_gradient = np.zeros_like(cn_l2_output)
        for i, cn_l2_neuron in enumerate(model.model["cn_layer_two"]):
            activation_derivative = algorithms.leaky_relu_gradient(cn_l2_neuron.activation)
            cn2_error_gradient[i] = algorithms.convolve_gradient(fc1_error_gradient[i], cn_l2_neuron.kernel) * activation_derivative

        cn1_error_gradient = np.zeros_like(cn_l1_output)
        for i, cn_l1_neuron in enumerate(model.model["cn_layer_one"]):
            activation_derivative = algorithms.leaky_relu_gradient(cn_l1_neuron.activation)
            cn1_error_gradient[i] = algorithms.convolve_gradient(cn2_error_gradient, cn_l1_neuron.kernel) * activation_derivative

        # Adjusting weights and biases
        for i, output_neuron in enumerate(model.model["output_layer"]):
            output_neuron.weights -= learning_rate * output_error_gradient[i] * np.array(fc_l2_output)
            output_neuron.bias -= learning_rate * output_error_gradient[i]

        for i, fc_l2_neuron in enumerate(model.model["fc_layer_two"]):
            fc_l2_neuron.weights -= learning_rate * fc2_error_gradient[i] * fc_l2_neuron.activation
            fc_l2_neuron.bias -= learning_rate * fc2_error_gradient[i]

        for i, fc_l1_neuron in enumerate(model.model["fc_layer_one"]):
            fc_l1_neuron.weights -= learning_rate * fc1_error_gradient[i] * fc_l1_neuron.activation
            fc_l1_neuron.bias -= learning_rate * fc1_error_gradient[i]

        for i, cn_l2_neuron in enumerate(model.model["cn_layer_two"]):
            cn_l2_neuron.kernel -= learning_rate * cn2_error_gradient[i] * cn_l2_neuron.activation
            cn_l2_neuron.bias -= learning_rate * cn2_error_gradient[i]

        for i, cn_l1_neuron in enumerate(model.model["cn_layer_one"]):
            cn_l1_neuron.kernel -= learning_rate * cn1_error_gradient[i] * cn_l1_neuron.activation
            cn_l1_neuron.bias -= learning_rate * cn1_error_gradient[i]

        

            

