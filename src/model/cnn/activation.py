import numpy as np

from maths.maths import EPSILON

""" 
    Activation Functions all make use of numpy broadcasting or fancy indexing
    rather than np.vectorize() which has a speed comparable to python loops
"""

def reLU(x):
    x[x < 0] = 0
    return x


def reLU_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0-sigmoid(x))


def softmax(x):
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    result = exps / np.sum(exps)
    return result



def softmax_derivative(x):
    # sm = softmax(x)
    # result = sm * (1-sm)
    return x


def activation_function(activation_key: str):
    activation_key = activation_key.lower()
    if activation_key == 'relu':
        return reLU
    elif activation_key == 'sigmoid':
        return sigmoid
    elif activation_key == 'softmax':
        return softmax
    else:
        raise Exception("No activation function found!")


def activation_derivative_function(activation_key: str):
    activation_key = activation_key.lower()
    if activation_key == 'relu':
        return reLU_derivative
    elif activation_key == 'sigmoid':
        return sigmoid_derivative
    elif activation_key == 'softmax':
        return softmax_derivative
    else:
        raise Exception("No activation derivative function found!")
