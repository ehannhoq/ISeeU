import json
import os
import numpy as np

import convolution_neuron as cn
import neuron

model = {}

<<<<<<< HEAD
input_size = (500, 500)

cn1_neurons = 16
cn1_kernel_shape = (3, 3)

cn2_neurons = 32
cn2_kernel_shape = (cn1_neurons, 3, 3)

fc1_neurons = 256
fc1_weights = 50

fc2_neurons = 128
output_neurons = 4


=======
>>>>>>> parent of bbc1e17 (switched from oop to matricies)
def save_data():
    global model
    with open(f'src/data.json', "w") as f:
        json.dump(serialize_data(model), f)


def load_data():
    global model
    with open(f'src/data.json', "r") as f:
        model = deserialize_data(json.load(f))

def serialize_data(model):
    return {
        "cn_layer_one": [neuron.kernel.tolist(), neuron.bias for neuron in model["cn_layer_one"]],
        "cn_layer_two": [neuron.kernel.tolist(), neuron.bias for neuron in model["cn_layer_two"]],
        "fc_layer_one": [(neuron.weights.tolist(), neuron.bias) for neuron in model["fc_layer_one"]],
        "fc_layer_two": [(neuron.weights.tolist(), neuron.bias) for neuron in model["fc_layer_two"]],
        "output_layer": [(neuron.weights.tolist(), neuron.bias) for neuron in model["output_layer"]]
    }

# Helper function to deserialize model
def deserialize_data(data):
    return {
        "cn_layer_one": [cn.ConvolutionNeuron(np.array(kernel), bias) for kernel, bias in data["cn_layer_one"]],
        "cn_layer_two": [cn.ConvolutionNeuron(np.array(kernel), bias) for kernel, bias in data["cn_layer_two"]],
        "fc_layer_one": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["fc_layer_one"]],
        "fc_layer_two": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["fc_layer_two"]],
        "output_layer": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["output_layer"]]
    }

    

def load_model():
    global model

    num_layer_one_cn = 16
    num_layer_two_cn = 32
    num_layer_one_fc = 256
    num_layer_two_fc = 128
    num_output_layer = 4


    if os.path.exists("src/data.json"):
        load_data()
    else:
        layer_one_cn = [ cn.ConvolutionNeuron(np.random.randn(3, 3), np.random.randn()) for _ in range(num_layer_one_cn) ]
        layer_two_cn = [ cn.ConvolutionNeuron(np.random.randn(num_layer_one_cn, 3, 3), np.random.randn()) for _ in range(num_layer_two_cn) ]
        
        layer_one_fc = [ neuron.Neuron(np.random.randn(50), np.random.randn()) for _ in range(num_layer_one_fc) ]
        layer_two_fc = [ neuron.Neuron(np.random.randn(num_layer_one_fc), np.random.randn()) for _ in range(num_layer_two_fc) ]
        
        output_layer = [ neuron.Neuron(np.random.randn(num_layer_two_fc), np.random.randn()) for _ in range(num_output_layer) ]


        model = {
            "cn_layer_one": layer_one_cn,
            "cn_layer_two": layer_two_cn,
            "fc_layer_one": layer_one_fc,
            "fc_layer_two": layer_two_fc,
            "output_layer": output_layer
        }

        save_data()

