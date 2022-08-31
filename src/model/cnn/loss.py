import numpy as np
from maths.maths import EPSILON


""" 
    Loss functions
"""

def mean_squared_error(x):
    return np.square(1.0 - x)

def cross_entropy(y, p):
    return -y*np.log(p)-(1-y)*np.log(1-p)

def cross_entropy_label_loss(y):
    return -np.log(y + EPSILON)

def cross_entropy_derivative(x, y, clip=True):
    result =  -(y/x - (1-y)/(1-x))
    return result
