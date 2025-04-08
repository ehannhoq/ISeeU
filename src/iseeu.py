import numpy as np

from algorithms import convolve
from algorithms import leaky_relu
from algorithms import max_pooling
from algorithms import sigmoid
from algorithms import assign_ground_truth
from algorithms import binary_cross_entropy_gradient
from algorithms import mse_gradient
from algorithms import leaky_relu_derivative
from algorithms import kernel_gradient
from algorithms import convolve_gradient


class Model:
    def __init__(self, learning_rate, confidence_threshold, stride, show_debug=False):
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.stride = stride
        self.show_debug = show_debug

        self.cn1_neurons = 16
        self.cn1_kernel_size = 3
        self.cn2_neurons = 32
        self.cn2_kernel_size = 3

        self.cn12_padding = self.cn2_kernel_size // 2
        
        self.cn3_neurons = 64
        self.cn3_kernel_size = 5
        self.cn4_neurons = 128
        self.cn4_kernel_size = 5

        self.cn34_padding = self.cn4_kernel_size // 2

        self.fc1_neurons = 256
        self.fc2_neurons = 128
        self.fc3_neurons = 64

        self.maximum_faces = 10
        self.output_neurons = self.maximum_faces * 5 # x, y, width, height, confidence

        input_height = 500
        input_width = 500
        exp_cn1_height = (input_height + 2 * self.cn12_padding - self.cn1_kernel_size) // stride + 1
        exp_cn1_width = (input_width + 2 * self.cn12_padding - self.cn1_kernel_size) // stride + 1

        exp_cn2_height = (exp_cn1_height + 2 * self.cn12_padding - self.cn2_kernel_size) // stride + 1
        exp_cn2_width = (exp_cn1_width + 2 * self.cn12_padding - self.cn2_kernel_size) // stride + 1

        exp_cn2_height //= 2
        exp_cn2_width //= 2

        exp_cn3_height = (exp_cn2_height + 2 * self.cn34_padding - self.cn3_kernel_size) // stride + 1
        exp_cn3_width = (exp_cn2_width + 2 * self.cn34_padding - self.cn3_kernel_size) // stride + 1

        exp_cn4_height = (exp_cn3_height + 2 * self.cn34_padding - self.cn4_kernel_size) // stride + 1
        exp_cn4_width = (exp_cn3_width + 2 * self.cn34_padding - self.cn4_kernel_size) // stride + 1

        exp_cn4_height //= 2
        exp_cn4_width //= 2

        exp_fc1_input = exp_cn4_width * exp_cn4_height * self.cn4_neurons
        
        self.cn1_w = np.random.uniform(-0.01, 0.01, (self.cn1_neurons, 1, self.cn1_kernel_size, self.cn1_kernel_size)).astype(np.float32)
        self.cn1_b = np.zeros((self.cn1_neurons,), dtype=np.float32)

        self.cn2_w = np.random.uniform(-0.01, 0.01, (self.cn2_neurons, self.cn1_neurons, self.cn2_kernel_size, self.cn2_kernel_size)).astype(np.float32)
        self.cn2_b = np.zeros((self.cn2_neurons,), dtype=np.float32)

        self.cn3_w = np.random.uniform(-0.01, 0.01, (self.cn3_neurons, self.cn2_neurons, self.cn3_kernel_size, self.cn3_kernel_size)).astype(np.float32)
        self.cn3_b = np.zeros((self.cn3_neurons,), dtype=np.float32)

        self.cn4_w = np.random.uniform(-0.01, 0.01, (self.cn4_neurons, self.cn3_neurons, self.cn4_kernel_size, self.cn4_kernel_size)).astype(np.float32)
        self.cn4_b = np.zeros((self.cn4_neurons,), dtype=np.float32)
        
        self.fc1_w = np.random.uniform(-0.01, 0.01, (self.fc1_neurons, exp_fc1_input)).astype(np.float32)
        self.fc1_b = np.zeros((self.fc1_neurons,), dtype=np.float32)

        self.fc2_w = np.random.uniform(-0.01, 0.01, (self.fc2_neurons, self.fc1_neurons)).astype(np.float32)    
        self.fc2_b = np.zeros((self.fc2_neurons,), dtype=np.float32)

        self.fc3_w = np.random.uniform(-0.01, 0.01, (self.fc3_neurons, self.fc2_neurons)).astype(np.float32)
        self.fc3_b = np.zeros((self.fc3_neurons,), dtype=np.float32)

        self.output_w = np.random.uniform(-0.01, 0.01, (self.output_neurons, self.fc3_neurons)).astype(np.float32)
        self.output_b = np.zeros((self.output_neurons,), dtype=np.float32)


    def train(self, input: np.ndarray, expected: np.ndarray):
        predicted = self.forward(input)
        self.backpropagation(input, predicted, expected)

    def forward(self, input: np.ndarray):
        batch_size = input.shape[0]

        cn1_z = convolve(
            input=input,
            filter=self.cn1_w,
            stride=self.stride,
            padding=self.cn12_padding
            )

        cn2_z = convolve(
            input=leaky_relu(cn1_z),
            filter=self.cn2_w,
            stride=self.stride,
            padding=self.cn12_padding
            )
        cn2_a = leaky_relu(cn2_z)
        cn2_a = max_pooling(cn2_a, pool_size=(2, 2))

        cn3_z = convolve(
            input=cn2_a,
            filter=self.cn3_w,
            stride=self.stride,
            padding=self.cn34_padding
            )

        cn4_z = convolve(
            input=leaky_relu(cn3_z),
            filter=self.cn4_w,
            stride=self.stride,
            padding=self.cn34_padding
            )
        
        cn4_a = leaky_relu(cn4_z)
        cn4_a = max_pooling(cn4_a, pool_size=(2, 2))

        flattened_cn_a4 = cn4_a.reshape(batch_size, -1)

        fc1_z = np.dot(flattened_cn_a4, self.fc1_w.T) + self.fc1_b

        fc2_z = np.dot(leaky_relu(fc1_z), self.fc2_w.T) + self.fc2_b

        fc3_z = np.dot(leaky_relu(fc2_z), self.fc3_w.T) + self.fc3_b

        output_z = np.dot(leaky_relu(fc3_z), self.output_w.T) + self.output_b
        output_z = np.reshape(output_z, (batch_size, self.maximum_faces, 5))

        bounding_boxes = np.reshape(leaky_relu(output_z[:, :, :4]), (batch_size, self.maximum_faces, 4))
        confidence = np.reshape(sigmoid(output_z[:, :, 4]), (batch_size, self.maximum_faces,))

        return cn1_z, cn2_z, cn3_z, cn4_z, fc1_z, fc2_z, fc3_z, output_z, (bounding_boxes, confidence)
    
    def backpropagation(self, input: np.ndarray, predictions:tuple, expected: np.ndarray):
        cn1_z, cn2_z, cn3_z, cn4_z, fc1_z, fc2_z, fc3_z, output_z, final_predictions = predictions
        bounding_boxes, confidence = final_predictions

        batch_size = cn1_z.shape[0]

        np.set_printoptions(precision=4, suppress=True)

        confidence_labels = assign_ground_truth(bounding_boxes, expected)
        confidence_gradient = binary_cross_entropy_gradient(confidence, confidence_labels, expected)

        bounding_boxes_gradient = mse_gradient(bounding_boxes, expected) * leaky_relu_derivative(output_z[:, :, :4])

        output_gradient = np.zeros_like(output_z)
        output_gradient[:, :, :4] = bounding_boxes_gradient
        output_gradient[:, :, 4] = confidence_gradient
        output_gradient = np.reshape(output_gradient, (batch_size, -1))
        output_w_gradient = np.dot(output_gradient.T, leaky_relu(fc3_z))


        fc3_gradient = np.dot(output_gradient, self.output_w) * leaky_relu_derivative(fc3_z)
        fc3_w_gradient = np.dot(fc3_gradient.T, leaky_relu(fc2_z))


        fc2_gradient = np.dot(fc3_gradient, self.fc3_w) * leaky_relu_derivative(fc2_z)
        fc2_w_gradient = np.dot(fc2_gradient.T, leaky_relu(fc1_z))


        fc1_gradient = np.dot(fc2_gradient, self.fc2_w) * leaky_relu_derivative(fc1_z)
        cn4_a = leaky_relu(cn4_z)
        cn4_a = max_pooling(cn4_a, pool_size=(2, 2))
        flattened_cn_a4 = cn4_a.reshape(batch_size, -1)
        fc1_w_gradient = np.dot(fc1_gradient.T, flattened_cn_a4)

        cn4_gradient = np.dot(fc1_gradient, self.fc1_w)
        cn4_gradient = np.reshape(cn4_gradient, max_pooling(input=cn4_z, pool_size=(2, 2)).shape) * leaky_relu_derivative(max_pooling(cn4_z, (2, 2)))
        cn4_w_gradient = kernel_gradient(cn3_z, cn4_gradient, self.cn4_w.shape)

        cn3_gradient = convolve_gradient(cn4_gradient, self.cn3_w, True)
        cn3_w_gradient = kernel_gradient(cn2_z, cn3_gradient, self.cn3_w.shape)


        cn2_gradient = convolve_gradient(cn3_gradient, self.cn2_w, False)
        cn2_w_gradient = kernel_gradient(cn1_z, cn2_gradient, self.cn2_w.shape)


        cn1_gradient = convolve_gradient(cn2_gradient, self.cn1_w, False)
        cn1_w_gradient = kernel_gradient(input, cn1_gradient, self.cn1_w.shape)

        if self.show_debug:
            print()
            print(f"Average gradient in CN1_W: {np.mean(cn1_w_gradient)}")
            print(f"Average gradient in CN2_W: {np.mean(cn2_w_gradient)}")
            print(f"Average gradient in CN3_W: {np.mean(cn3_w_gradient)}")
            print(f"Average gradient in CN4_W: {np.mean(cn4_w_gradient)}")
            print(f"Average gradient in FC1_W: {np.mean(fc1_w_gradient)}")
            print(f"Average gradient in FC2_W: {np.mean(fc2_w_gradient)}")
            print(f"Average gradient in FC3_W: {np.mean(fc3_w_gradient)}")
            print(f"Average gradient in OUTPUT_W: {np.mean(output_w_gradient)}")

            print(f"Average gradient in CN1_B: {np.mean(cn1_gradient)}")
            print(f"Average gradient in CN2_B: {np.mean(cn2_gradient)}")
            print(f"Average gradient in CN3_B: {np.mean(cn3_gradient)}")
            print(f"Average gradient in CN4_B: {np.mean(cn4_gradient)}")
            print(f"Average gradient in FC1_B: {np.mean(fc1_gradient)}")
            print(f"Average gradient in FC2_B: {np.mean(fc2_gradient)}")
            print(f"Average gradient in FC3_B: {np.mean(fc3_gradient)}")
            print(f"Average gradient in OUTPUT_B: {np.mean(output_gradient)}")
            print()            

        self.cn1_w -= self.learning_rate * cn1_w_gradient
        self.cn2_w -= self.learning_rate * cn2_w_gradient
        self.cn3_w -= self.learning_rate * cn3_w_gradient
        self.cn4_w -= self.learning_rate * cn4_w_gradient
        self.fc1_w -= self.learning_rate * fc1_w_gradient
        self.fc2_w -= self.learning_rate * fc2_w_gradient
        self.fc3_w -= self.learning_rate * fc3_w_gradient        
        self.output_w -= self.learning_rate * output_w_gradient

        self.cn1_b -= self.learning_rate * np.sum(cn1_gradient)
        self.cn2_b -= self.learning_rate * np.sum(cn2_gradient)
        self.cn3_b -= self.learning_rate * np.sum(cn3_gradient)
        self.cn4_b -= self.learning_rate * np.sum(cn4_gradient)
        self.fc1_b -= self.learning_rate * np.sum(fc1_gradient)
        self.fc2_b -= self.learning_rate * np.sum(fc2_gradient)
        self.fc3_b -= self.learning_rate * np.sum(fc3_gradient)
        self.output_b -= self.learning_rate * np.sum(output_gradient) 
