"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Abstract class of neural network layer
Description:

Abstract class for a layer that contains functions:
    - init
    - forward_propagation
    - backward_propagation

Parameters of the class
    - input
    - output
"""
class Layer:

    # Initisialases itself
    def __init__(self):
        self.input = None # Input of a layer X
        self.output = None #  Output of a layer Y

    # Computes an output for a layer
    def forward_propagation(self, input):
        raise NotImplementedError

    # Computes new values of the bias and weights in a layer.
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
