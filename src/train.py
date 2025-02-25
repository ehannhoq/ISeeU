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
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        print(f"\rLoading image {i}/{len(lines)}", end="")

        image_name = lines[i].strip()
        image_path = os.path.join(path, image_name)
        original_image = mpimg.imread(image_path)

        downscaled_image = original_image
        if len(downscaled_image.shape) == 3:
            downscaled_image = np.dot(downscaled_image[...,:3], [0.299, 0.587, 0.114])


        downscaled_image, aspect_ratio = algorithms.resize_image(image=downscaled_image, target_size=model.input_size)
        i += 1

        num_faces = int(lines[i].strip())
        i += 1

        if num_faces == 0:
            i += 1

        bounding_boxes = []
        for _ in range(num_faces):
            bbox_info = list(map(int, lines[i].split()))
            x, y, w, h = bbox_info[:4]
            bounding_boxes.append( (x, y, w, h) )
            i += 1

        dataset.append( {
            "original_image": original_image,
            "input_image": downscaled_image,
            "original_aspect_ratio": aspect_ratio, 
            "bounding_boxes": np.array(bounding_boxes)
            } )
    

    return dataset


if __name__ == '__main__':

    # TODO: Vectorizations for convolution layers' forward pass and backpropagation.
    # TODO: Transform model into object; better for reusability/modularity.
    # TODO: Add batching.

    # Fetching training data
    dataset = load_training_data('training_data')
    print("\nDataset loaded.")

    print("Loading model...")
    model.load_model(create_new_model=True)
    print("Model loaded.")

    display_predictions = False
    
    # Training
    for index, data in enumerate(dataset):

        # Preprosessing
        original_image = data["original_image"]
        input = data["input_image"]
        expected = data["bounding_boxes"]
        original_aspect_ratio = data["original_aspect_ratio"]
        learning_rate = 0.005

        # First Convolution Layer
        print("First CN layer...")
        cn1_activation = np.zeros((
            model.cn1_neurons,
            input.shape[0] - model.cn1_kernel_shape[0],
            input.shape[1] - model.cn1_kernel_shape[1]
        ))
        for i in range(model.cn1_neurons):
            cn1_activation[i] = algorithms.convolve2d(input, model.model["k_cn1"][i]) + model.model["b_cn1"][i]
        cn1_activation = algorithms.max_pooling(cn1_activation)
        print("First CN layer done.")

        # Second Convolution Layer
        print("Second CN layer...")
        cn2_activation = np.zeros((
            model.cn2_neurons,
            cn1_activation.shape[1] - model.cn2_kernel_shape[1],
            cn1_activation.shape[2] - model.cn2_kernel_shape[2]
        ))
        for i in range(model.cn2_neurons):
            cn2_activation[i] = algorithms.convolve3d(algorithms.leaky_relu(cn1_activation), model.model["k_cn2"][i]) + model.model["b_cn2"][i]
        cn2_activation = algorithms.max_pooling(cn2_activation)
        print("Second CN layer done.")

        # Flattened output for fully connected layer
        flattened_cn_output = algorithms.leaky_relu(cn2_activation).flatten()


        # First Fully Connected Layer
        print("First FC layer...")
        fc1_activation = np.dot(algorithms.leaky_relu(flattened_cn_output), model.model["w_fc1"]) + model.model["b_fc1"]
        print("First FC layer done.")

        # Second Fully Connected Layer
        print("Second FC layer...")
        fc2_activation = np.dot(algorithms.leaky_relu(fc1_activation), model.model["w_fc2"]) + model.model["b_fc2"]
        print("Second FC layer done.")


        # Output Layer
        print("Output layer...")
        fc2_activation_leaky = algorithms.leaky_relu(fc2_activation)         
        output_activation = np.dot(fc2_activation_leaky, model.model["w_output"]) + model.model["b_output"]
        output_activation = output_activation.reshape(model.maximum_faces, 5)
        print("Output layer done.")


        # Predictions
        output_bounding_boxes = algorithms.leaky_relu(output_activation[:, :4])
        output_confidence = algorithms.sigmoid(output_activation[:, 4])
        print("Prediction done.")
        
        if display_predictions:
            predictions = np.zeros_like(output_activation)
            predictions[:, :4] = output_bounding_boxes * original_aspect_ratio
            predictions[:, 4] = output_confidence
            print("Displaying prediction.")
            model.display_prediction(original_image, predictions)


        # Calculate error gradients
        print("Calculating output gradient...")
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
        print("Output gradients calculated.")


        print("Calculating FC2 gradient...")
        output_delta_flattened = output_delta.reshape(-1, 1).squeeze()
        fc2_delta = np.dot(output_delta_flattened, model.model["w_output"].T) * algorithms.leaky_relu_gradient(fc2_activation)
        fc2_delta = fc2_delta.squeeze()
        print("FC2 gradient calculated.")

        print("Calculating FC1 gradient...")
        fc1_delta = np.dot(fc2_delta, model.model["w_fc2"].T) * algorithms.leaky_relu_gradient(fc1_activation)
        fc1_delta = fc1_delta.squeeze()
        print("FC1 gradient calculated.")

        print("Calculating CN2 gradient...")
        cn2_delta_flattened = np.dot(fc1_delta, model.model["w_fc1"].T)
        cn2_delta = cn2_delta_flattened.reshape(cn2_activation.shape)
        cn2_delta *= algorithms.leaky_relu_gradient(cn2_activation)
        
        w_cn2_delta = np.zeros_like(model.model["k_cn2"])
        for i in range(model.cn2_neurons):
            for j in range(model.cn1_neurons):
                w_cn2_delta[i, j] = algorithms.compute_kernel_gradient(cn1_activation[j], cn2_delta[i], (5, 5))
        print("CN2 gradient calculated.")
        
        print("Calculating CN1 gradient...")
        cn1_delta = np.zeros_like(cn1_activation)
        for i in range(model.cn1_neurons):
            for j in range(model.cn2_neurons):
                cn1_delta[i] = algorithms.convolve_gradient(cn2_delta[j], np.flip(model.model["k_cn2"][j, i], axis=(0, 1)))
        cn1_delta *= algorithms.leaky_relu_gradient(cn1_activation)

        w_cn1_delta = np.zeros_like(model.model["k_cn1"])
        for i in range(model.cn1_neurons):
            w_cn1_delta[i] = algorithms.compute_kernel_gradient(input, cn1_delta[i], (3, 3))
        print("CN1 gradient calculated.")

        # Adjust weights and biases
        model.model["w_output"] -= learning_rate * np.dot(output_delta_flattened, output_activation.flatten())
        model.model["b_output"] -= learning_rate * np.sum(output_delta_flattened)

        model.model["w_fc2"] -= learning_rate * np.dot(fc2_delta, fc2_activation.flatten())
        model.model["b_fc2"] -= learning_rate * np.sum(fc2_delta)

        model.model["w_fc1"] -= learning_rate * np.dot(fc1_delta, fc1_activation.flatten())
        model.model["b_fc1"] -= learning_rate * np.sum(fc1_delta)

        model.model["k_cn2"] -= learning_rate * w_cn2_delta
        model.model["b_cn2"] -= learning_rate * np.sum(cn2_delta)

        model.model["k_cn1"] -= learning_rate * w_cn1_delta
        model.model["b_cn1"] -= learning_rate * np.sum(cn1_delta)
        print("Adjusted weights and biases.")


    model.save_data()






        