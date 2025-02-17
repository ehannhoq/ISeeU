import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import algorithms

import model

<<<<<<< HEAD
def load_images(path: str) -> np.ndarray:
=======
def load_images(path: str, grayscale: bool = True) -> np.array:
>>>>>>> parent of bbc1e17 (switched from oop to matricies)
    image_data = []

    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image = mpimg.imread(os.path.join(path, filename))

<<<<<<< HEAD
            aspect_ratio = image.shape[1] / image.shape[0]
            image = algorithms.resize_image(image=image, target_size=model.image_size)

            image_data.append(np.array([string_to_numbers(filename.split(".")[0])]), image, aspect_ratio)
=======
            if grayscale and len(image.shape) == 3:
                image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

            # For training and testing, we're going to downscale every image from A x B -> C x D, with either C or D
            # being 300 pixels, depending on whether A or B is greater. (ie. 600 x 400 -> 300 x 200, or 400 x 600 -> 200 x 300)
            image = algorithms.resize_image(image)


            image_data.append(np.array([string_to_numbers(filename.split(".")[0])]), image)
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

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

<<<<<<< HEAD
        # First Convolution Layer
        cn1_activation = np.zeros((
            model.cn1_neurons,
            image.shape[0] - model.cn1_kernel_shape[0],
            image.shape[1] - model.cn1_kernel_shape[1]
        ))
        for i in range(model.cn1_neurons):
            cn1_activation[i] = algorithms.convolve(image, model.model["w_cn1"][i]) + model.model["b_cn1"][i]
        cn1_activation = algorithms.max_pooling(cn1_activation)
=======
        # Applying non-linearity and pooling.
        cn_l1_output = algorithms.leaky_relu(cn_l1_output)
        cn_l1_output = algorithms.max_pooling(cn_l1_output)
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

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

<<<<<<< HEAD
        # Second Convolution Layer
        cn2_activation = np.zeros((
            model.cn2_neurons,
            cn1_activation.shape[0] - model.cn2_kernel_shape[0],
            cn1_activation.shape[1] - model.cn2_kernel_shape[1]
        ))
        for i in range(model.cn2_neurons):
            cn2_activation[i] = algorithms.convolve(algorithms.leaky_relu(cn1_activation), model.model["w_cn2"][i]) + model.model["b_cn2"][i]
        cn2_activation = algorithms.max_pooling(cn2_activation)
=======
        # Applying non-linearity and pooling.
        cn_l2_output = algorithms.leaky_relu(cn_l2_output)
        cn_l2_output = algorithms.max_pooling(cn_l2_output)
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

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

<<<<<<< HEAD
        # Second Fully Connected Layer
        fc2_activation = np.zeros((model.fc2_neurons))
        for i in range(model.fc2_neurons):
            fc2_activation[i] = np.dot(algorithms.leaky_relu(fc1_activation), model.model["w_fc2"][i]) + model.model["b_fc2"][i]

        output_activation = np.zeros((model.output_neurons))
        for i in range (model.output_neurons):
            output_activation[i] = np.dot(algorithms.leaky_relu(fc2_activation), model.model["w_output"]) + model.model["b_output"][i]
        output_activation = algorithms.leaky_relu(output_activation)

        pred_x = int( output_activation[0] )
        pred_y = int( output_activation[1] )
        pred_w = int( output_activation[2] )
        pred_h = int( output_activation[3] )

        pred_rect = patches.Rectangle(xy=(pred_x, pred_y), width=pred_w, height=pred_h, linewidth=2, edgecolor="r", facecolor="none")


        # Show prediction
        fig, axs = plt.subplot(1, 2, figsize=(12, 6))

        axs[0].imshow(image)
        axs[0].axis("off")

        axs[1].imshow(image)
        axs[1].add_patch(pred_rect)
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()


        # Calculate error gradients
        output_delta = algorithms.mean_squared_error_gradient(predicted=output_activation, expected=expected) * algorithms.leaky_relu_gradient(output_activation)
        output_delta = output_delta.reshape((-1, 1))

        fc2_delta = np.dot(model.model["w_fc2"].T, output_delta) * algorithms.leaky_relu_gradient(fc2_activation)
        fc2_delta = fc2_delta.reshape((-1, 1))

        fc1_delta = np.dot(model.model["w_fc1"].T, fc2_delta) * algorithms.leaky_relu_gradient(fc1_activation)
        fc1_delta = fc1_delta.reshape((-1, 1))

        fc1_delta_reshaped = fc1_delta.reshape(cn2_activation)
        cn2_delta = np.zeros_like(cn2_activation)
        for i in range(model.cn2_neurons):
            cn2_delta[i] = algorithms.gradient_convolve(fc1_delta_reshaped, model.model["k_cn2"][i]) * algorithms.leaky_relu_gradient(cn2_activation)

        cn1_delta = np.zeros_like(cn1_activation)
        for i in range(model.cn1_neurons):
            cn1_delta[i] = algorithms.gradient_convolve(fc2_delta, model.model["k_cn1"][i]) * algorithms.leaky_relu_gradient(cn1_activation)


        # Adjust weights/kernels and biases
        for i in range(model.output_neurons):
            model.model["w_output"][i] -= learning_rate * np.outer(fc2_activation, output_delta)
            model.model["b_output"][i] -= learning_rate * np.sum(output_delta)

        for i in range(model.fc2_neurons):
            model.model["w_fc2"][i] -= learning_rate * np.outer(fc1_activation, fc2_delta.T)
            model.model["b_fc2"][i] -= learning_rate * np.sum(fc2_delta)

        for i in range(model.fc1_neurons):
            model.model["w_fc1"][i] -= learning_rate * np.outer(fc2_activation, fc1_delta.T)
            model.model["b_fc1"][i] -= learning_rate * np.sum(fc1_delta)

        for i in range(model.cn2_neurons):
            model.model["k_cn2"][i] -= learning_rate * np.dot(cn1_activation, cn2_delta.T)
            model.model["b_cn2"][i] -= learning_rate * np.sum(cn2_delta)

        for i in range(model.cn1_neurons):
            model.model["k_cn1"][i] -= learning_rate * np.dot(image, cn1_delta.T)
            model.model["b_cn1"][i] -= learning_rate * np.sum(cn1_delta)



=======
        predicted_x = output[0].activation
        predicted_y = output[1].activation
        predicted_width = output[2].activation
        predicted_height = output[3].activation
>>>>>>> parent of bbc1e17 (switched from oop to matricies)

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
            cn2_error_gradient[i] = cn_l2_neuron.convolve_gradient(fc1_error_gradient[i]) * activation_derivative

        cn1_error_gradient = np.zeros_like(cn_l1_output)
        for i, cn_l1_neuron in enumerate(model.model["cn_layer_one"]):
            activation_derivative = algorithms.leaky_relu_gradient(cn_l1_neuron.activation)
            cn1_error_gradient[i] = cn_l1_neuron.convolve_gradient(cn2_error_gradient) * activation_derivative

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

        

            

