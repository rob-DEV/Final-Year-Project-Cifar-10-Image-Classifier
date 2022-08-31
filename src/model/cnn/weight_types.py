import numpy as np


def random_uniform_weights(weight_shape):
    weights = np.array(np.random.uniform(-1.0, 1.0, weight_shape) * 0.1, dtype=np.float64)
    return weights.astype(np.float64)

def xavier_uniform_weights(weight_shape, input_units=1, output_units=1):
    weight_limit = np.sqrt(6.0 / (input_units + output_units)) 
    weights = np.array(np.random.uniform(-weight_limit, weight_limit, weight_shape), dtype=np.float64)
    return weights