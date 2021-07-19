# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# cd Desktop/Dissertation/python/Neural\ Network/scratch/MyNN/
import numpy as np
from random import randint

from neural_network_files.neuralNetwork import NeuralNetwork
from neural_network_files.preactiveLayer import PreactiveLayer
from neural_network_files.activeLayer import ActiveLayer
from neural_network_files.activationFunctions import sigmoid, sigmoid_prime, hyperbolic, hyperbolic_prime
from neural_network_files.costFunctions import mse, mse_prime
from neural_network_files.saveModel import save_model, load_model
from research_networks.graphs import plot_error

error = []

# network
net = NeuralNetwork()
net.add_layer(PreactiveLayer(2, 2))
net.add_layer(ActiveLayer(hyperbolic, hyperbolic_prime))
net.add_layer(PreactiveLayer(2, 1))
net.add_layer(ActiveLayer(hyperbolic, hyperbolic_prime))
net.set_cost_function(mse, mse_prime)

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

print(x_train.shape)

x_test = np.array([[[0,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]],[[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]],[[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[0,1]], [[1,0]],[[1,1]]])
y_test = np.array([[[0]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]],[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]],[[0]]])

error.append(net.fit(x_train, y_train, epochs=2000, learning_rate=0.01, type="sgd"))
# validate

plot_error(error)
net.score_function(x_test, y_test)
