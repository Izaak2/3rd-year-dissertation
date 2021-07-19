"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Activation Function for a single neuron
Description: Mathematical equations implemented into Python
"""

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def hyperbolic(x):
    return np.tanh(x)

def hyperbolic_prime(x):
    return 1 - np.tanh(x)**2
