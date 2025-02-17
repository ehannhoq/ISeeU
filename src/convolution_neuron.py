import numpy as np
import algorithms


class ConvolutionNeuron:
    kernel: np.ndarray

    activation: np.ndarray
    bias: float

    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def convolve2d(self, input: np.ndarray):

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_height, input_width = input.shape
        kernel_height, kernel_width = self.kernel.shape

        output_width = input_width - kernel_width
        output_height = input_height - kernel_height
        self.activation = np.zeros((output_height, output_width), dtype=float)

        for y in range(output_height):
            for x in range(output_width):
                window = input[y:y + kernel_height, x:x + kernel_width]
                self.activation[y, x] = np.sum(window * self.kernel)
        
        self.activation += self.bias

    def convolve3d(self, input: np.ndarray):

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_depth, input_height, input_width = input.shape
        kernel_depth, kernel_height, kernel_width = self.kernel.shape

        assert kernel_depth == input_depth, "Kernel and input shape mismatch"

        output_width = input_width - kernel_width
        output_height = input_height - kernel_height

        self.activation = np.zeros((output_height, output_width), dtype=float)


        for y in range(output_height):
            for x in range(output_width):
                cube = input[:, y:y + kernel_height, x:x + kernel_width]
                self.activation[y, x] = np.sum(cube * self.kernel)