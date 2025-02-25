import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import algorithms

class ISeeU:
    def __init__(self, learning_rate:float, confidence_threshold:float, show_debug:bool = False):
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
        self.maximum_faces = 25 
        self.output_neurons = 5 * self.maximum_faces

        if os.path.exists("src/model_weights.npz"):
            debug_message("Loading weights...", self.show_debug)
            self.load_weights()
            debug_message("Weights loaded.", self.show_debug)
        else:
            debug_message("Generating new model...", self.show_debug)
            self.new_model()
            debug_message("Model generated.", self.show_debug)

    def new_model(self):
        k_cn1 = np.random.randn(self.cn1_neurons, self.cn1_kernel_shape[0], self.cn1_kernel_shape[1])
        k_cn2 = np.random.randn(self.cn2_neurons, self.cn2_kernel_shape[0], self.cn2_kernel_shape[1], self.cn2_kernel_shape[2])

        output_height_cn1 = self.input_size[0] - self.cn1_kernel_shape[0]
        output_width_cn1 = self.input_size[1] - self.cn1_kernel_shape[0]

        output_height_cn1 = output_height_cn1 // 2
        output_width_cn1 = output_width_cn1 // 2

        output_height_cn2 = output_height_cn1 - self.cn2_kernel_shape[1]
        output_width_cn2 = output_width_cn1 - self.cn2_kernel_shape[2]

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
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            
            ax.text(x=x, y=y, s=f"Confidence: {confidence:.2f}", color="red", fontsize=12, ha="center", va="center")
        plt.show()

    def downscale_image(self, image:np.ndarray):

        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        downscaled_image, aspect_ratio = algorithms.resize_image(image=image, target_size=self.input_size)

        return (downscaled_image, aspect_ratio)

    
    def train(self, dataset:list):
        for i, data in enumerate(dataset):
            image, aspect_ratio = self.downscale_image(data["image"])
            expected = data["bounding_boxes"]

            activations = self.forward_pass(image)
            self.backpropagation(image, activations, expected)

            debug_message(f"Image {i + 1} of {len(dataset)} processed.", self.show_debug)

        self.save_weights()



    def inference(self, input:np.ndarray):
        downscaled_image, original_aspect_ratio = self.downscale_image(input)
        predictions = self.forward_pass(downscaled_image, inference=True)
        predictions[:, :4] *= original_aspect_ratio
        self.display_prediction(image=input, predictions=predictions)


    def forward_pass(self, input, inference:bool = False):
        # First Convolution Layer
        debug_message("First CN layer...", self.show_debug)
        cn1_activation = np.zeros((
            self.cn1_neurons,
            input.shape[0] - self.cn1_kernel_shape[0],
            input.shape[1] - self.cn1_kernel_shape[1]
        ))
        for i in range(self.cn1_neurons):
            cn1_activation[i] = algorithms.convolve2d(input, self.model["k_cn1"][i]) + self.model["b_cn1"][i]
        cn1_activation = algorithms.max_pooling(cn1_activation)
        debug_message("First CN layer done.", self.show_debug)

        # Second Convolution Layer
        debug_message("Second CN layer...", self.show_debug)
        cn2_activation = np.zeros((
            self.cn2_neurons,
            cn1_activation.shape[1] - self.cn2_kernel_shape[1],
            cn1_activation.shape[2] - self.cn2_kernel_shape[2]
        ))
        for i in range(self.cn2_neurons):
            cn2_activation[i] = algorithms.convolve3d(algorithms.leaky_relu(cn1_activation), self.model["k_cn2"][i]) + self.model["b_cn2"][i]
        cn2_activation = algorithms.max_pooling(cn2_activation)
        debug_message("Second CN layer done.", self.show_debug)

        # Flattened output for fully connected layer
        flattened_cn_output = algorithms.leaky_relu(cn2_activation).flatten()


        # First Fully Connected Layer
        debug_message("First FC layer...", self.show_debug)
        fc1_activation = np.dot(algorithms.leaky_relu(flattened_cn_output), self.model["w_fc1"]) + self.model["b_fc1"]
        debug_message("First FC layer done.", self.show_debug)

        # Second Fully Connected Layer
        debug_message("Second FC layer...", self.show_debug)
        fc2_activation = np.dot(algorithms.leaky_relu(fc1_activation), self.model["w_fc2"]) + self.model["b_fc2"]
        debug_message("Second FC layer done.", self.show_debug)


        # Output Layer
        debug_message("Output layer...", self.show_debug)
        fc2_activation_leaky = algorithms.leaky_relu(fc2_activation)         
        output_activation = np.dot(fc2_activation_leaky, self.model["w_output"]) + self.model["b_output"]
        output_activation = output_activation.reshape(self.maximum_faces, 5)
        debug_message("Output layer done.", self.show_debug)


        # Predictions
        output_bounding_boxes = algorithms.leaky_relu(output_activation[:, :4])
        output_confidence = algorithms.sigmoid(output_activation[:, 4])

        if not inference:
            return (cn1_activation, cn2_activation, fc1_activation, fc2_activation, output_activation, (output_bounding_boxes, output_confidence))

        valid_predictions = output_confidence >= self.confidence_threshold
        bbox_predictions = output_bounding_boxes[valid_predictions]
        conf_predictions = output_confidence[valid_predictions]

        predictions = np.hstack((bbox_predictions, conf_predictions[:, np.newaxis]))
        return predictions



    def backpropagation(self, input, activations, expected):
        cn1_activation, cn2_activation, fc1_activation, fc2_activation, output_activation, predictions = activations
        output_bounding_boxes, output_confidence = predictions

        # Calculate error gradients
        debug_message("Calculating output gradient...", self.show_debug)
        confidence_labels = algorithms.assign_ground_truth(output_bounding_boxes, expected)
        confidence_delta = algorithms.binary_cross_entropy_gradient(confidence_labels, output_confidence)

        bounding_box_delta = np.zeros_like(output_bounding_boxes)
        for index, box in enumerate(output_bounding_boxes):
            if confidence_labels[index] == 0: continue
            bounding_box_delta[index] = (
                algorithms.mean_squared_error_gradient(output_bounding_boxes[index], expected[index])
                * algorithms.leaky_relu_gradient(output_activation[index, :4])
            )
        if np.sum(confidence_labels) == 0:
            bounding_box_delta = 1e-4 * np.ones_like(output_bounding_boxes)

        output_delta = np.zeros_like(output_activation)
        output_delta[:, :4] = bounding_box_delta
        output_delta[:, 4] = confidence_delta
        debug_message("Output gradient done.", self.show_debug)


        debug_message("Calculating FC2 gradient...", self.show_debug)
        output_delta_flattened = output_delta.reshape(-1, 1).squeeze()
        fc2_delta = np.dot(output_delta_flattened, self.model["w_output"].T) * algorithms.leaky_relu_gradient(fc2_activation)
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
        
        w_cn2_delta = np.zeros_like(self.model["k_cn2"])
        for i in range(self.cn2_neurons):
            for j in range(self.cn1_neurons):
                w_cn2_delta[i, j] = algorithms.compute_kernel_gradient(cn1_activation[j], cn2_delta[i], (5, 5))

        debug_message("CN2 gradient done.", self.show_debug)
        

        debug_message("Calculating CN1 gradient...", self.show_debug)
        cn1_delta = np.zeros_like(cn1_activation)
        for i in range(self.cn1_neurons):
            for j in range(self.cn2_neurons):
                cn1_delta[i] = algorithms.convolve_gradient(cn2_delta[j], np.flip(self.model["k_cn2"][j, i], axis=(0, 1)))
        cn1_delta *= algorithms.leaky_relu_gradient(cn1_activation)

        w_cn1_delta = np.zeros_like(self.model["k_cn1"])
        for i in range(self.cn1_neurons):
            w_cn1_delta[i] = algorithms.compute_kernel_gradient(input, cn1_delta[i], (3, 3))
        
        debug_message("CN1 gradient done.", self.show_debug)

        # Adjust weights and biases
        self.model["w_output"] -= self.learning_rate * np.dot(output_delta_flattened, output_activation.flatten())
        self.model["b_output"] -= self.learning_rate * np.sum(output_delta_flattened)

        self.model["w_fc2"] -= self.learning_rate * np.dot(fc2_delta, fc2_activation.flatten())
        self.model["b_fc2"] -= self.learning_rate * np.sum(fc2_delta)

        self.model["w_fc1"] -= self.learning_rate * np.dot(fc1_delta, fc1_activation.flatten())
        self.model["b_fc1"] -= self.learning_rate * np.sum(fc1_delta)

        self.model["k_cn2"] -= self.learning_rate * w_cn2_delta
        self.model["b_cn2"] -= self.learning_rate * np.sum(cn2_delta)

        self.model["k_cn1"] -= self.learning_rate * w_cn1_delta
        self.model["b_cn1"] -= self.learning_rate * np.sum(cn1_delta)

        debug_message("Weights adjusted.", self.show_debug)


def debug_message(message:str, show:bool):
    if show:
        print(message)


