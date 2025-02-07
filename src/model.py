import json
import os
import numpy as np

import convolution_neuron as cn
import neuron

model = {}

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
        "cn_layer_one": [neuron.kernel.tolist() for neuron in model["cn_layer_one"]],
        "cn_layer_two": [neuron.kernel.tolist() for neuron in model["cn_layer_two"]],
        "fc_layer_one": [(neuron.weights.tolist(), neuron.bias) for neuron in model["fc_layer_one"]],
        "fc_layer_two": [(neuron.weights.tolist(), neuron.bias) for neuron in model["fc_layer_two"]],
        "output_neurons": [(neuron.weights.tolist(), neuron.bias) for neuron in model["output_neurons"]]
    }

# Helper function to deserialize model
def deserialize_data(data):
    return {
        "cn_layer_one": [cn.ConvolutionNeuron(np.array(kernel)) for kernel in data["cn_layer_one"]],
        "cn_layer_two": [cn.ConvolutionNeuron(np.array(kernel)) for kernel in data["cn_layer_two"]],
        "fc_layer_one": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["fc_layer_one"]],
        "fc_layer_two": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["fc_layer_two"]],
        "output_neurons": [neuron.Neuron(np.array(weights), bias) for weights, bias in data["output_neurons"]]
    }

    

def load_model():
    global model

    num_layer_one_cn = 16
    num_layer_two_cn = 32
    num_layer_one_fc = 256
    num_layer_two_fc = 128
    num_output_neurons = 4


    if os.path.exists("src/data.json"):
        load_data()
    else:
        layer_one_cn = [ cn.ConvolutionNeuron(np.random.randn(3, 3)) for _ in range(num_layer_one_cn) ]
        layer_two_cn = [ cn.ConvolutionNeuron(np.random.randn(3, 3)) for _ in range(num_layer_two_cn) ]
        layer_one_fc = [ neuron.Neuron(np.random.randn(num_layer_one_fc), np.random.randn()) for _ in range(num_layer_one_fc) ]
        layer_two_fc = [ neuron.Neuron(np.random.randn(num_layer_two_fc), np.random.randn()) for _ in range(num_layer_two_fc) ]
        output_neurons = [ neuron.Neuron(np.random.randn(num_layer_two_fc), np.random.randn()) for _ in range(num_output_neurons) ]


        model = {
            "cn_layer_one": layer_one_cn,
            "cn_layer_two": layer_two_cn,
            "fc_layer_one": layer_one_fc,
            "fc_layer_two": layer_two_fc,
            "output_neurons": output_neurons
        }

        save_data()

