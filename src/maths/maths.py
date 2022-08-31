import numpy as np

EPSILON = 1e-8

def euclidean_distance(a, b):
    """
    Calculates the euclidean distance between 2 1xN vectors.
    Parameters:
        a (np.ndarray): A one demensional vector.
        b (np.ndarray): A one demensional vector.
    Returns:
        result (np.ndarray): The distance between the vectors.
    """
    return np.sqrt(np.sum(np.square(a-b)))


def vectorized_euclidean_distance(a, b):
    """
    Calculates the euclidean distance between 2 NxN vectors.
    Parameters:
        a (np.ndarray): An N demensional vector.
        b (np.ndarray): An N demensional vector.
    Returns:
        result (np.ndarray): The distance between the vectors.
    """
    sum_of_squares_of_b = np.sum(np.square(b), axis=1, keepdims=True)
    sum_of_squares_of_a = np.sum(np.square(a), axis=1, keepdims=True)

    dists = np.sqrt(-2 * b.dot(a.T) + sum_of_squares_of_b + sum_of_squares_of_a.T)
    return dists


def next_multiple(n: int, m: int):
    """
    Calculates the next, closest value to n which is a multiple of m.
    Parameters:
        n (int): A number
        m (int): An N demensional vector.
    Returns:
        result (int): The next higher number which is a multiple of m.
    """
    if m == 0:
        return n
    remainder = n % m
    if remainder == 0:
        return n
    return n + m - remainder