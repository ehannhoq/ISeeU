import numpy as np
import matplotlib.image as mpimg

import model
import algorithms

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    image = mpimg.imread(image_path)

    # Preprosessing
    if len(image.shape) == 3:
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])

    downscaled_image, aspect_ratio = algorithms.resize_image(image=image, target_size=model.input_size)

    model.load_model(create_new_model=False)
    print("Model loaded.")
    
    confidence_threshold = 0.8


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
    
    predictions = np.zeros_like(output_activation)
    predictions[:, :4] = output_bounding_boxes * aspect_ratio
    predictions[:, 4] = output_confidence
    print("Displaying prediction.")
    model.display_prediction(image, predictions)
    