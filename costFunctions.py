"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Cost Functions for a single neuron
Description: Mathematical equations implemented into Python
"""


import numpy as np

def mse(true_output, predicted_output):
   return np.mean(np.power(true_output-predicted_output, 2))

def mse_prime(true_output, predicted_output):
   return 2*(predicted_output-true_output)/true_output.size
