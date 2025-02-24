import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


model = {}

input_size = (500, 500)

cn1_neurons = 16
cn1_kernel_shape = (3, 3)

cn2_neurons = 32
cn2_kernel_shape = (cn1_neurons, 5, 5)

fc1_neurons = 256
fc2_neurons = 128

maximum_faces = 25
output_neurons = 5 * maximum_faces


def save_data():
    global model
    np.savez_compressed("src/model_weights.npz", **model)


def load_data():
    global model
    loaded = np.load("src/model_weights.npz", allow_pickle=True)
    model.update(loaded)
        
def new_model():
    global model

    k_cn1 = np.random.randn(cn1_neurons, cn1_kernel_shape[0], cn1_kernel_shape[1])
    k_cn2 = np.random.randn(cn2_neurons, cn2_kernel_shape[0], cn2_kernel_shape[1], cn2_kernel_shape[2])

    output_height_cn1 = input_size[0] - cn1_kernel_shape[0]
    output_width_cn1 = input_size[1] - cn1_kernel_shape[0]

    output_height_cn1 = output_height_cn1 // 2
    output_width_cn1 = output_width_cn1 // 2

    output_height_cn2 = output_height_cn1 - cn2_kernel_shape[1]
    output_width_cn2 = output_width_cn1 - cn2_kernel_shape[2]

    output_height_cn2 = output_height_cn2 // 2
    output_width_cn2 = output_width_cn2 // 2

    flattened_size = cn2_neurons * output_height_cn2 * output_width_cn2
    
    w_fc1 = np.random.randn(flattened_size, fc1_neurons)
    w_fc2 = np.random.randn(fc1_neurons, fc2_neurons)   

    w_output = np.random.randn(fc2_neurons, output_neurons)

    b_cn1 = np.random.randn(cn1_neurons)
    b_cn2 = np.random.randn(cn2_neurons)
    b_fc1 = np.random.randn(fc1_neurons)
    b_fc2 = np.random.randn(fc2_neurons)
    b_output = np.random.randn(output_neurons)
    

    model = {
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

    save_data()


def load_model(create_new_model: bool):
    global model

    if os.path.exists("src/model_weights.npz"):
        load_data()
    elif create_new_model:
        new_model()
    else:
        raise Exception("Model not found.")


def display_prediction(image, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for (x, y, w, h, confidence) in predictions:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        
        ax.text(x=x, y=y, s=f"Confidence: {confidence:.2f}", color="red", fontsize=12, ha="center", va="center")
    plt.show()