import unittest

import numpy as np

from model.cnn.activation import reLU, reLU_derivative, sigmoid, sigmoid_derivative, softmax, activation_function, activation_derivative_function


class ActivationFunctionsTests(unittest.TestCase):

    def test_relu(self):
        input = np.array([-1, -2, -3, 0, 1, 2, 3, 4])
        expected = np.array([0, 0, 0, 0, 1, 2, 3, 4])

        result = reLU(input)

        np.testing.assert_equal(expected, result)

    def test_relu_der(self):
        input = np.array([-1, -2, -3, 0, 1, 2, 3, 4])
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = reLU_derivative(input)

        np.testing.assert_equal(expected, result)

    def test_sigmoid(self):
        input = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0.1192029, 0.2689414, 0.5, 0.7310586, 0.8807971])

        result = sigmoid(input)

        np.testing.assert_allclose(expected, result, rtol=0.01)

    def test_sigmoid_der(self):
        input = np.array([-2])
        expected = sigmoid(input) * (1.0-sigmoid(input))

        result = sigmoid_derivative(input)

        np.testing.assert_allclose(expected, result, rtol=0.01)

    def test_softmax(self):
        input = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        result = softmax(input)

        np.testing.assert_equal(1, np.sum(result))


if __name__ == '__main__':
    unittest.main()
