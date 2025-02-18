import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import algorithms

import model

def load_training_data(path: str):
    dataset = []

    with open(os.path.join(path, "image_info.txt"), mode='r') as f:
        lines = f.readlines

    i = 0
    while i < len(lines):
        image_name = lines[i].strip()
        image_path = os.path.join(path, image_name)
        image = mpimg.imread(image_path)
        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        aspect_ratio = image.shape[1] / image.shape[0]
        dataset["aspect_ratios"].append(aspect_ratio)
        image = algorithms.resize_image(image=image, target_size=model.image_size)
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        bounding_boxes = []
        for _ in range(num_faces):
            bbox_info = list(map(int, lines[i].split()))
            x, y, w, h = bbox_info[:4]
            bounding_boxes.append( (x, y, w, h) )
            i += 1

        dataset.append( {
            "image": image, 
            "original_aspect_ratio": aspect_ratio, 
            "bounding_boxes": bounding_boxes
            } )
    
        return dataset



if __name__ == '__main__':
    # Fetching training data
    dataset = load_training_data('training_data')

    model.load_model()
    
    # Training
    for index, data in enumerate(dataset):

        # Preprosessing
        image = data["image"]
        expected = data["bounding_boxes"]
        original_aspect_ratio = data["original_aspect_ratio"]
        learning_rate = 0.005

        # First Convolution Layer
        cn1_activation = np.zeros((
            model.cn1_neurons,
            image.shape[0] - model.cn1_kernel_shape[0],
            image.shape[1] - model.cn1_kernel_shape[1]
        ))
        for i in range(model.cn1_neurons):
            cn1_activation[i] = algorithms.convolve2d(image, model.model["w_cn1"][i]) + model.model["b_cn1"][i]
        cn1_activation = algorithms.max_pooling(cn1_activation)


        # Second Convolution Layer
        cn2_activation = np.zeros((
            model.cn2_neurons,
            cn1_activation.shape[0] - model.cn2_kernel_shape[0],
            cn1_activation.shape[1] - model.cn2_kernel_shape[1]
        ))
        for i in range(model.cn2_neurons):
            cn2_activation[i] = algorithms.convolve3d(algorithms.leaky_relu(cn1_activation), model.model["w_cn2"][i]) + model.model["b_cn2"][i]
        cn2_activation = algorithms.max_pooling(cn2_activation)


        # Flattened output for fully connected layer
        flattened_cn_output = algorithms.leaky_relu(cn2_activation).flatten()


        # First Fully Connected Layer
        fc1_activation = np.zeros((model.fc1_neurons))
        for i in range(model.fc1_neurons):
            fc1_activation[i] = np.dot(flattened_cn_output, model.model["w_fc1"][i]) + model.model["b_fc1"][i]
        

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
            cn2_delta[i] = algorithms.convolve_gradient(fc1_delta_reshaped, model.model["k_cn2"][i]) * algorithms.leaky_relu_gradient(cn2_activation)

        cn1_delta = np.zeros_like(cn1_activation)
        for i in range(model.cn1_neurons):
            cn1_delta[i] = algorithms.convolve_gradient(fc2_delta, model.model["k_cn1"][i]) * algorithms.leaky_relu_gradient(cn1_activation)


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

    
    model.save_data()






        
        
        

            

