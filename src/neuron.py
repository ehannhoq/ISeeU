import numpy as np

class Neuron:
    weights: np.ndarray
    bias: float

    activation: float

    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def correction(target: float):
        pass