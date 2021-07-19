"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Neural Netowork class
Description: Contains all of the functions needed to manage a whole neural network.

Contains
    __init__(self)
    add_layer(self, layer)
    set_cost_function(self, cost_function, cost_function_prime)
    predict_output(self, input)
    fit(self, x_train, y_train, epochs, learning_rate, mini_batch_size=32, type="sgd", print_error=True)
    __fit_mini_batching(self, x_train, y_train, epochs, learning_rate, mini_batch_size=32, print_error=True)
    __fit_sgd(self, x_train, y_train, epochs, learning_rate,print_error=True)
    __forward_propagation(self, input):
    __backward_propagation(self, error, learning_rate)
    score_function(self, x_test, y_test)
    score_function__biggest_one(self, x_test, y_test)
"""

from random import randint
import numpy as np

"""
    This class manages all of the network tasks as training and predicting
"""
class NeuralNetwork:
    """
    Initisialases itsself with a list of layers and cost_function and cost_function prime.
    """
    def __init__(self):
        self.layers = []
        self.cost_function = None
        self.cost_function_prime = None

    """
    Adding layers to the list.
    The layers must be in order that
    1st is preactive layer
    2nd is active layer
    no limi to the number of layers
    """
    def add_layer(self, layer):
        self.layers.append(layer)


    """
    Set cost_function functuion for the network
    """
    def set_cost_function(self, cost_function, cost_function_prime):
        self.cost_function = cost_function
        self.cost_function_prime = cost_function_prime

    """
    Predicts output for a given input
    """
    def predict_output(self, input):
        # Create result array that will be returned
        result = []
        # Go through each sample
        for sample in range(len(input)):
            # Compute forward propagation through each layer in
            # the network
            net_output = self.__forward_propagation(input[sample])
            result.append(net_output)
        return result

    """
    sgd - stochastic gradient descent
    mb - mini batching
    """
    def fit(self, x_train, y_train, epochs, learning_rate, mini_batch_size=32, type="sgd", print_error=True):
        print("Training started")
        if type == "sgd":
            error = self.__fit_sgd(x_train, y_train, epochs, learning_rate, print_error=print_error)
            return error
        elif type == "mb":
            error = self.__fit_mini_batching(x_train, y_train, epochs, learning_rate, mini_batch_size, print_error=print_error)
            return error

    """
    Creatingin mini batch for the network
    // Simpelst way I could thought about mini batching. not efficient.
    // not testes properly.
    """
    def __fit_mini_batching(self, x_train, y_train, epochs, learning_rate, mini_batch_size=32, print_error=True):
        iterations = len(x_train)
        graph_error = []
        for i in range(epochs):
            disp_err = 0
            randomize = np.arange(len(x_train))
            np.random.shuffle(randomize)
            x_train = x_train[randomize]
            y_train = y_train[randomize]
            error = 0

            for iteration in range(iterations):
                net_output = self.__forward_propagation(x_train[iteration])
                error = np.add(error, self.cost_function_prime(y_train[iteration], net_output))
                if ((iteration+1) % mini_batch_size) == 0:
                    error /= mini_batch_size
                    self.__backward_propagation(error, learning_rate)
                    error = 0
                    disp_err += self.cost_function(y_train[iteration], net_output)

            if print_error:
                print('epoch %d/%d   error=%f' % (i+1, epochs, disp_err/(len(x_train)//mini_batch_size)))
            graph_error.append(disp_err)
        return graph_error

    """
    Training netwrok with given set of data using stochastic gradient descent
    """
    def __fit_sgd(self, x_train, y_train, epochs, learning_rate,print_error=True):
        iterations = len(x_train)
        graph_error = []
        # training loop
        for epoch in range(epochs):
            disp_err = 0

            for iteration in range(iterations):
                # Computes network output using forward propagation
                net_output = self.__forward_propagation(x_train[iteration])
                # For displaying purposes gets all of the errors summed
                disp_err += self.cost_function(y_train[iteration], net_output)
                # Backward propagation
                self.__backward_propagation(self.cost_function_prime(y_train[iteration], net_output), learning_rate)
            # calculate average error on all samples
            disp_err /= iterations
            graph_error.append(disp_err)
            if print_error:
                print('epoch %d/%d   error=%f' % (epoch+1, epochs, disp_err))
        return graph_error

        """
        # forward propagation
        Getting first input for a layer and the the output becomes input
        input = x_train[sample]
        for layer in self.layers:
            output = layer.forward_propagation(input)
            input = output
        """

    def __forward_propagation(self, input):
        # gets output for each layer and eventualy net_output
        for layer in self.layers:
            input = layer.forward_propagation(input)
        return input

    def __backward_propagation(self, error, learning_rate):
        # gets the input error of the network by getting input of each layer
        # updates each layer paramteres with respect to the output error in
        # each layer
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

    def score_function(self, x_test, y_test):
        iterations = len(x_test)
        good = 0;
        for i in range(iterations):
            #print(y_test[i])
            out = self.predict_output(x_test[i])
            #print(out)
            out = np.around(out)
            out = out.reshape(1, out.shape[2])
            out = out.astype(int)
            x, y = out.shape
            ideal = np.empty([1,y])
            ideal[0] = y_test[i]

            if np.array_equal(ideal , out):
                good += 1
        print("All samples ", iterations, " and correct answers ", good)
        print("Accuarcy is ", (good/iterations)*100, "%")

    def score_function__biggest_one(self, x_test, y_test):
        iterations = len(x_test)

        good = 0
        for i in range(iterations):
            out = self.predict_output(x_test[i])
            test_length = len(out[0][0])
            max = np.amax(out)
            for j in range(test_length):
                if out[0][0][j] == max:
                    out[0][0][j] = 1
                else:
                    out[0][0][j] = 0

            if np.array_equal(y_test[i] , out[0][0]):
                good += 1
            #else:
                #print(y_test[i])
        print("All samples ", iterations, " and correct answers ", good)
        print("Accuarcy is ", round((good/iterations)*100,2), "%")
        return (good/iterations)*100
