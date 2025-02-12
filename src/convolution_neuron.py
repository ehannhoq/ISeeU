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

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_height, input_width = input.shape
        kernel_height, kernel_width = self.kernel.shape

        output_width = (input_width - kernel_width) // self.stride + 1
        output_height = (input_height - kernel_height) // self.stride + 1
        self.activation = np.zeros((output_height, output_width), dtype=float)

        for x in range(0, output_width, self.stride):
            for y in range(0, output_height, self.stride):
                window = np.array( input[y:y + len(self.kernel), x:x + len(self.kernel[0])] )
                self.activation[y, x] = np.sum(window * self.kernel)

        if nonLinear:
            activation = algorithms.leaky_relu(activation)

        if pooling:
            self.activation = algorithms.max_pooling(activation)

    def convolve3d(self, input: np.ndarray, nonLinear:bool = False, pooling = False):

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_depth, input_height, input_width = input.shape
        kernel_depth, kernel_height, kernel_width = self.kernel.shape

        assert kernel_depth == input_depth, "Kernel and input shape mismatch"

        output_width = (input_width - kernel_width) // self.stride + 1
        output_height = (input_height - kernel_height) // self.stride + 1

        self.activation = np.zeros((output_height, output_width), dtype=float)


        for x in range(0, output_width, self.stride):
            for y in range(0, output_height, self.stride):
                cube = np.array( input[:, y:y + kernel_height, x:x + kernel_width] )
                self.activation[y, x] = np.sum(cube * self.kernel)

        if nonLinear:
            self.activation = algorithms.leaky_relu(self.activation)

        if pooling:
            self.activation = algorithms.max_pooling(self.activation)
        

    def correction(target: np.ndarray):
        pass