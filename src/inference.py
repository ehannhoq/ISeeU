import numpy as np
import matplotlib.image as mpimg

import model
import algorithms

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    image = mpimg.imread(image_path)

    if len(image.shape) == 3:
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

    aspect_ratio = image.shape[1] / image.shape[0]
    image = algorithms.resize_image(image=image, target_size=model.input_size)
    image = np.expand_dims(image, axis=0 )

    model.load_model(create_new_model=False)
    print("Model loaded.")
    
    # Preprosessing
    confidence_threshold = 0.8

    # First Convolution Layer
    cn1_activation = np.zeros((
        model.cn1_neurons,
        image.shape[0] - model.cn1_kernel_shape[0],
        image.shape[1] - model.cn1_kernel_shape[1]
    ))
    for i in range(model.cn1_neurons):
        cn1_activation[i] = algorithms.convolve2d(image, model.model["k_cn1"][i]) + model.model["b_cn1"][i]
    cn1_activation = algorithms.max_pooling(cn1_activation)
    print("First CN layer done.")

    # Second Convolution Layer
    cn2_activation = np.zeros((
        model.cn2_neurons,
        cn1_activation.shape[0] - model.cn2_kernel_shape[0],
        cn1_activation.shape[1] - model.cn2_kernel_shape[1]
    ))
    for i in range(model.cn2_neurons):
        cn2_activation[i] = algorithms.convolve3d(algorithms.leaky_relu(cn1_activation), model.model["k_cn2"][i]) + model.model["b_cn2"][i]
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
    print("Output layer done.")

    predictions = []
    for i in range(len(output_confidence)):
        if output_confidence[i] > confidence_threshold:
            predictions.append(output_bounding_boxes[i] * aspect_ratio)
    print("Predictions done. Displaying...")

    model.display_prediction(image, predictions)
    