"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Active part of a single layer
Description: This file helps to organise and manage an active part of a layer in Neural Network
"""


from .layer import Layer
import numpy as np
"""
    Inheritance from Layer class

"""
class ActiveLayer(Layer):
    """
    Sets active_function and its derivative to the layer
    Any active function can be implemented
    """
    def __init__(self, active_function, active_function_prime):
        self.active_function = active_function
        self.active_function_prime = active_function_prime

    """
    Computing forward propagation for given input for active part of a layer
    """
    def forward_propagation(self, input):
        self.input = input
        self.output = self.active_function(input)
        return self.output

    """
    Returns derivative of error E with respect to the input

    learing rate is not used due to lacness of learnable parameters
    """
    def backward_propagation(self, output_error, learning_rate):
        return self.active_function_prime(self.input) * output_error
