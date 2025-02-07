import numpy as np
import algorithms


class ConvolutionNeuron:
    kernel: np.ndarray
    stride: int

    activation: np.ndarray

    def __init__(self, kernel: np.ndarray, stride: int = 1):
        self.kernel = kernel
        self.stride = stride

    def convolve2d(self, input: np.ndarray, nonLinear:bool = False, pooling = False):
        output_rows: int = len(input) - len(self.kernel) + 1
        output_cols: int = len(input[0]) - len(self.kernel[0]) + 1
        activation = np.zeros((output_rows, output_cols), dtype=float)

        for x in range(0, output_rows, self.stride):
            for y in range(0, output_cols, self.stride):
                window = np.array([ input[i][y:y + len(self.kernel[0])] for i in range(x, x + len(self.kernel)) ])
                activation[x, y] = np.sum(window * self.kernel)

        if nonLinear:
            activation = algorithms.leaky_relu(activation)

        if pooling:
            activation = algorithms.max_pooling(activation)


    def correction(target: np.ndarray):
        pass