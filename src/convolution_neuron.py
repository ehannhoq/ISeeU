import numpy as np
import algorithms


class ConvolutionNeuron:
    kernel: np.ndarray
    stride: int

    activation: np.ndarray

    def __init__(self, kernel: np.ndarray, stride: int = 1):
        self.kernel = kernel
        self.stride = stride

    def convolve2d(self, input: np.ndarray):

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_height, input_width = input.shape
        kernel_height, kernel_width = self.kernel.shape

        output_width = (input_width - kernel_width) // self.stride + 1
        output_height = (input_height - kernel_height) // self.stride + 1
        self.activation = np.zeros((output_height, output_width), dtype=float)

        for y in range(0, output_height, self.stride):
            for x in range(0, output_width, self.stride):
                window = input[y:y + kernel_height, x:x + kernel_width]
                self.activation[y, x] = np.sum(window * self.kernel)

    def convolve3d(self, input: np.ndarray):

        assert len(self.kernel.shape) == len(input.shape), "Kernel and input dimension mismatch"

        input_depth, input_height, input_width = input.shape
        kernel_depth, kernel_height, kernel_width = self.kernel.shape

        assert kernel_depth == input_depth, "Kernel and input shape mismatch"

        output_width = (input_width - kernel_width) // self.stride + 1
        output_height = (input_height - kernel_height) // self.stride + 1

        self.activation = np.zeros((output_height, output_width), dtype=float)


        for y in range(0, output_height, self.stride):
            for x in range(0, output_width, self.stride):
                cube = input[:, y:y + kernel_height, x:x + kernel_width]
                self.activation[y, x] = np.sum(cube * self.kernel)

    def correction(target: np.ndarray):
        pass