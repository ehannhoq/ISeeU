import numpy as np

class Neuron:
    weights: np.ndarray
    bias: float

    activation: float

    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def calculate_activation(self, input: np.ndarray):
        self.activation = np.dot(input, self.activation) + self.bias