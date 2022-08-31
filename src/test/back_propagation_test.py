import unittest

import numpy as np

from model.logistic_regression.lr import LogisticRegression

class BackpropagationTests(unittest.TestCase):

    def test_logistic_regression_back_propagation(self):
        # Create a logistic regression model with a mock dataset
        x_train = np.random.randn(100, 10)
        y_train = np.random.randint(0,1, 100)

        lr = LogisticRegression(learning_rate=0.01, num_iterations=10)
        lr._init_weights_bias(x_train)

        epsilon = 1e-7
        weights = lr.w
        forward = lr._forward(x_train)

        weights_plus = weights + epsilon 
        weights_minus = weights - epsilon 

        lr.w = weights_plus
        forward_plus = lr._forward(x_train)

        # lr.w = weights_minus
        # forward_minus = lr._forward(x_train)

        # gradient_approx = (forward_plus - forward_minus) / (2 * epsilon)

        # gradients = x_train

        # numerator = np.linalg.norm(gradients - gradient_approx)
        # denominator = np.linalg.norm(gradients) + np.linalg.norm(gradient_approx)
        # difference = numerator/ denominator


if __name__ == '__main__':
    unittest.main()
