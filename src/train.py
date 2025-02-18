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
        image_name = lines[i].strip()
        image_path = os.path.join(path, image_name)
        image = mpimg.imread(image_path)
        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        aspect_ratio = image.shape[1] / image.shape[0]
        image = algorithms.resize_image(image=image, target_size=model.input_size)
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
            "bounding_boxes": np.array(bounding_boxes)
            } )
    
        return dataset
    
def display_prediction(image, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for (x, y, w, h, confidence) in predictions:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        ax.text((x, y), f"Confidence: {confidence:.2f}", color="red", fontsize=12, ha="center", va="center")

    plt.show()



if __name__ == '__main__':
    # Fetching training data
    print("Loading training dataset!")
    dataset = load_training_data('training_data')
    print("Dataset loaded.")

    print("Loading model!")
    model.load_model()
    print("Model loaded.")
    
    # Training
    for index, data in enumerate(dataset):

        # Preprosessing
        image = data["image"]
        expected = data["bounding_boxes"]
        original_aspect_ratio = data["original_aspect_ratio"]

        confidence_threshold = 0.8
        learning_rate = 0.005

        print("Preprosessng done.")

        # First Convolution Layer
        cn1_activation = np.zeros((
            model.cn1_neurons,
            image.shape[0] - model.cn1_kernel_shape[0],
            image.shape[1] - model.cn1_kernel_shape[1]
        ))
        for i in range(model.cn1_neurons):
            cn1_activation[i] = algorithms.convolve2d(image, model.model["w_cn1"][i]) + model.model["b_cn1"][i]
        cn1_activation = algorithms.max_pooling(cn1_activation)

        print("First CN layer done.")

        # Second Convolution Layer
        cn2_activation = np.zeros((
            model.cn2_neurons,
            cn1_activation.shape[0] - model.cn2_kernel_shape[0],
            cn1_activation.shape[1] - model.cn2_kernel_shape[1]
        ))
        for i in range(model.cn2_neurons):
            cn2_activation[i] = algorithms.convolve3d(algorithms.leaky_relu(cn1_activation), model.model["w_cn2"][i]) + model.model["b_cn2"][i]
        cn2_activation = algorithms.max_pooling(cn2_activation)

        print("Second CN layer done.")

        # Flattened output for fully connected layer
        flattened_cn_output = algorithms.leaky_relu(cn2_activation).flatten()


        # First Fully Connected Layer
        fc1_activation = np.dot(algorithms.leaky_relu(flattened_cn_output), model.model["w_fc1"]) + model.model["b_fc1"]
        print("First FC layer done.")

        # Second Fully Connected Layer
        fc2_activation = np.dot(algorithms.leaky_relu(fc1_activation), model.model["w_fc2"]) + model.model["b_fc2"]
        print("Second FC layer done.")


        output_activation = np.zeros((
            model.output_neurons,
            5 # x, y, w, h, confidence
            ))
        
        fc2_activation_leaky = algorithms.leaky_relu(fc2_activation)
        for i in range(model.output_neurons):
            output_activation[i] = np.dot(fc2_activation_leaky, model.model["w_output"]) + model.model["b_output"][i]

        output_bounding_boxes = algorithms.leaky_relu(output_activation[:, :4])
        output_confidence = algorithms.sigmoid(output_activation[:, 4])

        display_prediction(image, algorithms.leaky_relu(output_activation))

        confidence_labels = algorithms.assign_ground_truth(output_bounding_boxes, expected)

        # Calculate error gradients
        confidence_delta = algorithms.binary_cross_entropy_gradient(confidence_labels, output_confidence)

        for index, box in enumerate(output_bounding_boxes):
            if confidence_labels[index] == 0: continue
            bounding_box_delta = algorithms.mean_squared_error_gradient(output_bounding_boxes, expected) * algorithms.leaky_relu_gradient(output_activation[:, :4])

        fc2_delta = np.dot(model.model["w_fc2"].T, bounding_box_delta) * algorithms.leaky_relu_gradient(fc2_activation)
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

        # Adjust weights and biases
        model.model["w_output"][:, :4] -= learning_rate * np.dot(fc2_activation, bounding_box_delta.T)
        model.model["b_outout"][:, :4] -= learning_rate * np.sum(bounding_box_delta, acis=0)

        model.model["w_output"][:, 4] -= learning_rate * np.dot(fc2_activation, confidence_delta)
        model.model["b_output"][:, 4] -= learning_rate * np.sum(confidence_delta, acis=0)

        model.model["w_fc2"] -= learning_rate * np.dot(fc1_activation, fc2_delta.T)
        model.model["b_fc2"] -= learning_rate * np.sum(fc2_delta, acis=0)

        model.model["w_cn2"] -= learning_rate * np.dot(cn1_activation, cn2_delta.T)
        model.model["b_cn2"] -= learning_rate * np.sum(cn2_delta, acis=0)

        model.model["w_cn1"] -= learning_rate * np.dot(image, cn1_delta.T)
        model.model["b_cn1"] -= learning_rate * np.sum(cn1_delta, acis=0)

    model.save_data()






        
        
        

            

