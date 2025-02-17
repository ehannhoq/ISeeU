import json
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
output_neurons = 4


def save_data():
    global model
    with open(f'src/data.json', "w") as f:
        json.dump(serialize_data(model), f)
        

def load_data():
    global model
    with open(f'src/data.json', "r") as f:
        model = deserialize_data(json.load(f))
        

def serialize_data(data):
    return {
        "w_cn1": [ cn1_weights.tolist() for cn1_weights in data["w_cn1"] ],
        "w_cn2": [ cn2_weights.tolist() for cn2_weights in data["w_cn2"] ],
        "w_fc1": [ fc1_weights.tolist() for fc1_weights in data["w_fc1"] ],
        "w_fc2": [ fc2_weights.tolist() for fc2_weights in data["w_fc2"] ],
        "w_output": [ output_weights.tolist() for output_weights in data["w_output"] ],

        "b_cn1": [ cn1_bias.tolist() for cn1_bias in data["b_cn1"] ],
        "b_cn2": [ cn2_bias.tolist() for cn2_bias in data["b_cn2"] ],
        "b_fc1": [ fc1_bias.tolist() for fc1_bias in data["b_fc1"] ],
        "b_fc2": [ fc2_bias.tolist() for fc2_bias in data["b_fc2"] ],
        "b_output": [ output_bias.tolist() for output_bias in data["b_output"] ]
    }


def deserialize_data(data):
    return {
        "w_cn1": np.array(data["w_cn1"]),
        "w_cn2": np.array(data["w_cn2"]),
        "w_fc1": np.array(data["w_fc1"]),
        "w_fc2": np.array(data["w_fc2"]),
        "w_output": np.array(data["w_output"]),

        "b_cn1": np.array(data["b_cn1"]),
        "b_cn2": np.array(data["b_cn2"]),
        "b_fc1": np.array(data["b_fc1"]),
        "b_fc2": np.array(data["b_fc2"]),
        "b_output": np.array(data["b_output"])
    }


def load_model():
    global model

    if os.path.exists("src/data.json"):
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
