"""
Author: Kacper Izaak Wojtasik
Data: 7/6/21
Project Title: Voice Recognition with Neural Network
File Title: Stroing traing model of Neural Network
Description: Saving model as a picle type that can be reloaded later.
"""

import pickle

def save_model(model,filename):
    filename = filename
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    filename = filename
    with open(filename, 'rb') as file:
        return pickle.load(file)
