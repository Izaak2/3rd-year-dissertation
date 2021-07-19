"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Preactive layer par for a single neuron
Description: Mathematical equations implemented into Python
"""

from .layer import Layer
import numpy as np

"""
That class is inheriting from layer class

The purpose of this class is to perfrom all of the tasks needed in a preactive stage of a layer
"""

class PreactiveLayer(Layer):
    """
    Initisialsing its weights and biases with random values from a uniform distribution over [-0.5, 0.5].
    It is very importnat to substract 0.5 coz when the value is close to 1
    the neuron could struggle to learn due to activation function graph

    Takes:
        - number of inputs
        - number of outputs

    Returns:
        - nothing
    """
    def __init__(self, input_size, output_size):
        # Uniformly generating random weights and biases in the range of -0.5 to 0.5.
        self.bias = (np.random.rand(1, output_size) - 0.5)
        self.weight = (np.random.rand(input_size, output_size) - 0.5)

    """
    Copmute forward propagation for a neuron with gven data
    Takes:
        - Input input_data
    Returns:
        - output of a neuron Y

    Y = X * W + B
    where:
        X is input data matrix
        W is a weight matrix
        B is a bias matrix
        Y is a ouput matrix
    """
    def forward_propagation(self, input):
        # Assigining input to the layer input and then computes output from actual parameters
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    """
    Compute backward propagation for a neuron with given error E and learning rate
    Returns input error dE/dY

    Input error equation
    dE      dE
    -   =   -   *   transpose(W)
    dX      dY

    Derivative of output error with respect to weights
    dE                      dE
    -   =   transpose(X) *  -
    dW                      dY

    Derivative of output error with respect to biases
    dE      dE
    -   =   -
    dX      dY

    Equation for updating parameters
    paramter = parameter - (learning_rate * parameter error)
    """
    def backward_propagation(self, output_error, learning_rate):
        # Computes input error
        input_error = self.input_error(output_error)

        # Updating weights and biases
        self.update_parameters(self.weight_gradient(output_error), self.bias_gradient(output_error), learning_rate)
        return input_error

    def input_error(self, output_error):
        return np.dot(output_error, self.weight.T)

    def weight_gradient(self, output_error):
        return np.dot(self.input.T, output_error)

    def bias_gradient(self, output_error):
        return output_error

    def update_parameters(self, weight_gradient, bias_gradient, learning_rate):
        self.weight = self.weight - (learning_rate * weight_gradient)
        self.bias = self.bias - (learning_rate * bias_gradient)
