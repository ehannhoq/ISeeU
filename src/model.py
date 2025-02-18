import os
import numpy as np


model = {}

input_size = (500, 500)

cn1_neurons = 16
cn1_kernel_shape = (3, 3)

cn2_neurons = 32
cn2_kernel_shape = (cn1_neurons, 3, 3)

fc1_neurons = 256
fc1_weights = 50

fc2_neurons = 128

maximum_faces = 100
output_neurons = 5 * maximum_faces


def save_data():
    global model
    np.savez_compressed("src/model.npz", **model)


def load_data():
    global model
    loaded = np.load("src/model.npz", allow_pickle=True)
    model.update(loaded)
        

def load_model():
    global model

    if os.path.exists("src/model.npz"):
        load_data()
    else:
        k_cn1 = np.random.randn(cn1_neurons, cn1_kernel_shape[0], cn1_kernel_shape[1])
        k_cn2 = np.random.randn(cn2_neurons, cn2_kernel_shape[0], cn2_kernel_shape[1], cn2_kernel_shape[2])

        output_height_cn1 = input_size[0] - cn1_kernel_shape[0]
        output_width_cn1 = input_size[1] - cn1_kernel_shape[0]

        output_height_cn2 = output_height_cn1 - cn2_kernel_shape[0]
        output_width_cn2 = output_width_cn1 - cn2_kernel_shape[1]

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
