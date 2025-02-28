import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import algorithms

class ISeeU:
    def __init__(self, confidence_threshold:float, learning_rate:float = 0.005, show_debug:bool = False):
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.show_debug = show_debug

        self.model = {}
        self.input_size = (500, 500)
        self.cn1_neurons = 16
        self.cn1_kernel_shape = (3, 3)
        self.cn2_neurons = 32
        self.cn2_kernel_shape = (self.cn1_neurons, 5, 5)
        self.fc1_neurons = 256
        self.fc2_neurons = 128
        self.maximum_faces = 10
        self.output_neurons = 5 * self.maximum_faces

        self.convPadding = 1
        self.convStride = 1

        if os.path.exists("src/model_weights.npz"):
            debug_message("Loading weights...", self.show_debug)
            self.load_weights()
            debug_message("Weights loaded.", self.show_debug)
        else:
            debug_message("Generating new weights...", self.show_debug)
            self.new_weights()
            debug_message("Weights generated.", self.show_debug)

    def new_weights(self):
        k_cn1 = np.random.randn(self.cn1_neurons, 1, self.cn1_kernel_shape[0], self.cn1_kernel_shape[1])
        k_cn2 = np.random.randn(self.cn2_neurons, self.cn2_kernel_shape[0], self.cn2_kernel_shape[1], self.cn2_kernel_shape[2])

        output_height_cn1 = (self.input_size[0] + 2 * self.convPadding - self.cn1_kernel_shape[0]) // self.convStride + 1
        output_width_cn1 = (self.input_size[1] + 2 * self.convPadding - self.cn1_kernel_shape[0]) // self.convStride + 1

        output_height_cn1 = output_height_cn1 // 2
        output_width_cn1 = output_width_cn1 // 2

        output_height_cn2 = (output_height_cn1 + 2 * (self.convPadding + 1) - self.cn2_kernel_shape[1]) // self.convStride + 1
        output_width_cn2 = (output_width_cn1 + 2 * (self.convPadding + 1) - self.cn2_kernel_shape[2]) // self.convStride + 1

        output_height_cn2 = output_height_cn2 // 2
        output_width_cn2 = output_width_cn2 // 2

        flattened_size = self.cn2_neurons * output_height_cn2 * output_width_cn2
        
        w_fc1 = np.random.randn(flattened_size, self.fc1_neurons)
        w_fc2 = np.random.randn(self.fc1_neurons, self.fc2_neurons)   

        w_output = np.random.randn(self.fc2_neurons, self.output_neurons)

        b_cn1 = np.random.randn(self.cn1_neurons)
        b_cn2 = np.random.randn(self.cn2_neurons)
        b_fc1 = np.random.randn(self.fc1_neurons)
        b_fc2 = np.random.randn(self.fc2_neurons)
        b_output = np.random.randn(self.output_neurons)

        self.model = {
            "k_cn1": k_cn1,
            "k_cn2": k_cn2,
            "w_fc1": w_fc1,
            "w_fc2": w_fc2,
            "w_output": w_output,

            "b_cn1": b_cn1,
            "b_cn2": b_cn2,
            "b_fc1": b_fc1,
            "b_fc2": b_fc2,
            "b_output": b_output
        }

        self.save_weights()

    def save_weights(self):
        np.savez_compressed("src/model_weights.npz", **self.model)


    def load_weights(self):
        loaded = np.load("src/model_weights.npz", allow_pickle=True)
        self.model = {key: loaded[key] for key in loaded.files}


    def display_prediction(self, image:np.ndarray, predictions:np.ndarray):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for (x, y, w, h, confidence) in predictions:
            print(x, y, w, h, confidence)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            
            ax.text(x=x, y=y, s=f"Confidence: {confidence:.2f}", color="black", fontsize=12, ha="center", va="center")
        plt.show()
    

    def train(self, input:np.ndarray, expected:list):
        activations = self.forward_pass(input)
        self.backpropagation(activations, expected, input)
        self.save_weights()


    def inference(self, input:np.ndarray):
        image = input

        if len(input.shape) == 3:
            input = np.dot(input[...,:3], [0.299, 0.587, 0.114])
        input, aspect_ratio = algorithms.resize_image(image=input, target_size=self.input_size)
        input = np.reshape(input, (1, 1, input.shape[0], input.shape[1]))

        predictions = self.forward_pass(input, inference=True)
        predictions[:, :4] *= aspect_ratio
        print(predictions)
        self.display_prediction(image=image, predictions=predictions)


    def forward_pass(self, input, inference:bool = False):
        # First Convolution Layer
        debug_message("First CN layer...", self.show_debug)
        cn1_activation = algorithms.convolve(input, self.model["k_cn1"], padding=self.convPadding, stride=self.convStride)
        cn1_activation = algorithms.max_pooling(cn1_activation)
        debug_message("First CN layer done.", self.show_debug)

        # Second Convolution Layer
        debug_message("Second CN layer...", self.show_debug)
        cn2_activation = algorithms.convolve(cn1_activation, self.model["k_cn2"], padding=self.convPadding + 1, stride=self.convStride)
        cn2_activation = algorithms.max_pooling(cn2_activation)

        # Flattened output for fully connected layer
        flattened_cn_output = algorithms.leaky_relu(cn2_activation)
        flattened_cn_output = np.reshape(flattened_cn_output, (cn2_activation.shape[0], -1))
        debug_message("Second CN layer done.", self.show_debug)

        # First Fully Connected Layer
        debug_message("First FC layer...", self.show_debug)
        fc1_activation = np.dot(flattened_cn_output, self.model["w_fc1"]) + self.model["b_fc1"]
        debug_message("First FC layer done.", self.show_debug)


        # Second Fully Connected Layer
        debug_message("Second FC layer...", self.show_debug)
        fc2_activation = np.dot(algorithms.leaky_relu(fc1_activation), self.model["w_fc2"]) + self.model["b_fc2"]
        debug_message("Second FC layer done.", self.show_debug)


        # Output Layer
        debug_message("Output layer...", self.show_debug)
        output_activation = np.dot(algorithms.leaky_relu(fc2_activation), self.model["w_output"]) + self.model["b_output"]
        output_activation = np.reshape(output_activation, (output_activation.shape[0], self.maximum_faces, 5))
        debug_message("Output layer done.", self.show_debug)


        # Predictions
        output_bounding_boxes = algorithms.leaky_relu(output_activation[:, :, :4])
        output_confidence = algorithms.sigmoid(output_activation[:, :, 4])

        if not inference:
            return (cn1_activation, cn2_activation, fc1_activation, fc2_activation, output_activation, (output_bounding_boxes, output_confidence))


        valid_predictions = output_confidence[0] >= self.confidence_threshold
        bbox_predictions = output_bounding_boxes[0, valid_predictions]
        conf_predictions = output_confidence[0, valid_predictions]

        predictions = np.hstack((bbox_predictions, conf_predictions[:, np.newaxis]))
        return predictions



    def backpropagation(self, activations:np.ndarray, expected:list, input:np.ndarray):
        cn1_activation, cn2_activation, fc1_activation, fc2_activation, output_activation, predictions = activations
        output_bounding_boxes, output_confidence = predictions

        batch_size = input.shape[0]
        
        # Calculate error gradients
        debug_message("Calculating output gradient...", self.show_debug)
        confidence_labels = algorithms.assign_ground_truth(output_bounding_boxes, expected)
        confidence_delta = algorithms.binary_cross_entropy_gradient(confidence_labels, output_confidence)

        bounding_box_delta = algorithms.mean_squared_error_gradient(output_bounding_boxes, expected) * algorithms.leaky_relu(output_activation[:, :, :4])

        if np.sum(confidence_labels) == 0:
            bounding_box_delta = 1e-4 * np.ones_like(output_bounding_boxes)

        output_delta = np.zeros_like(output_activation)
        output_delta[:, :, :4] = bounding_box_delta
        output_delta[:, :, 4] = confidence_delta
        debug_message("Output gradient done.", self.show_debug)


        debug_message("Calculating FC2 gradient...", self.show_debug)
        output_delta = np.reshape(output_delta, (batch_size, -1))
        fc2_delta = np.dot(output_delta, self.model["w_output"].T) * algorithms.leaky_relu_gradient(fc2_activation)
        fc2_delta = fc2_delta.squeeze()
        debug_message("FC2 gradient done.", self.show_debug)


        debug_message("Calculating FC1 gradient...", self.show_debug)
        fc1_delta = np.dot(fc2_delta, self.model["w_fc2"].T) * algorithms.leaky_relu_gradient(fc1_activation)
        fc1_delta = fc1_delta.squeeze()
        debug_message("FC1 gradient done.", self.show_debug)


        debug_message("Calculating CN2 gradient...", self.show_debug)
        cn2_delta_flattened = np.dot(fc1_delta, self.model["w_fc1"].T)
        cn2_delta = cn2_delta_flattened.reshape(cn2_activation.shape)
        cn2_delta *= algorithms.leaky_relu_gradient(cn2_activation)
        w_cn2_delta = algorithms.kernel_gradient(cn1_activation, cn2_delta, self.model["k_cn2"].shape)
        debug_message("CN2 gradient done.", self.show_debug)


        debug_message("Calculating CN1 gradient...", self.show_debug)
        cn1_delta = algorithms.convolve_gradient(cn2_delta, np.flip(self.model["k_cn2"], axis=(2, 3)))
        w_cn1_delta = algorithms.kernel_gradient(input, cn1_delta, self.model["k_cn1"].shape)
        debug_message("CN1 gradient done.", self.show_debug)
        

        # Adjust weights and biases

        for b in range(batch_size):
            self.model["w_output"] -= self.learning_rate * np.dot(output_delta[b], output_activation[b].flatten())
            self.model["b_output"] -= self.learning_rate * np.sum(output_delta[b])

            self.model["w_fc2"] -= self.learning_rate * np.dot(fc2_delta[b], fc2_activation[b].flatten())
            self.model["b_fc2"] -= self.learning_rate * np.sum(fc2_delta[b])

            self.model["w_fc1"] -= self.learning_rate * np.dot(fc1_delta[b], fc1_activation[b].flatten())
            self.model["b_fc1"] -= self.learning_rate * np.sum(fc1_delta[b])

        self.model["k_cn2"] -= self.learning_rate * w_cn2_delta
        self.model["b_cn2"] -= self.learning_rate * np.sum(cn2_delta)

        self.model["k_cn1"] -= self.learning_rate * w_cn1_delta
        self.model["b_cn1"] -= self.learning_rate * np.sum(cn1_delta)

            
        debug_message("Weights adjusted.", self.show_debug)


def debug_message(message:str, show:bool):
    if show:
        print(message)


