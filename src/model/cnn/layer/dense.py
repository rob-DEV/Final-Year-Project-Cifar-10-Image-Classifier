from ast import Tuple
import numpy as np
from model.cnn.layer.layer import Layer
from model.cnn.weight_types import random_uniform_weights, xavier_uniform_weights


class Dense(Layer):
    """
    Dense layer, a fully connected layer used for regression, forward feeds the previous layer output to the output units and applies a activation to them.
    """

    def __init__(self, name="", activation="relu", output_units=1, input_shape:Tuple=None, weight_init_type='xavier_uniform', clip_value=1) -> None:
        """
        Initializes a dense layer with the standard name, activation along with the input units from the previous layer and the desired number of outputs.
            name (str): Mainly used for debugging and also printed in summarize.
            activation (str): Activation string to determine the activation function and its derivative (see activation.py)
            output_units (int): Number of inputs in this layer, corresponds to the flattened number of outputs in the previous layer.
            input_shape (Tuple): Number of desired outputs for the layer
        """
        super().__init__(name, activation)
        self.input_shape = input_shape
        self.output_units = output_units

        if weight_init_type == 'xavier_uniform':
            self.weight_init_type = 'xavier_uniform'
        else:
            self.weight_init_type = 'random_uniform'

        self.clip_value = clip_value
            

    def compile(self, previous_layer):
        """
        Called during the compilation of a model. This method provides a way for the network to calculate the shapes for the input and output of the layer.
        """
        if previous_layer is None:
            # Ensure this layer (the first) has an input shape and determine the output shape
            if self.input_shape is None:
                raise Exception("No input shape has been provided for the first layer of the network!")
        else:
            if self.input_shape is None and previous_layer.output_shape is None:
                raise Exception("Unable to determine the input shape as the previous layer has no defined output shape")
            if self.input_shape is None:
                self.input_shape = previous_layer.output_shape

        self.output_shape = (1, self.output_units)

        # Setup a weight matrix from previous layer feature map dims
        if self.weight_init_type == 'xavier_uniform':
            self.w = xavier_uniform_weights(weight_shape=(np.prod(self.input_shape), self.output_units), input_units=np.prod(self.input_shape))
        else:
            self.w = random_uniform_weights(weight_shape=(np.prod(self.input_shape), self.output_units))
            

        self.b = np.zeros((1, self.output_units))

    def forward(self, input: np.ndarray):
        """
        Forward pass through dense. Takes a feature map of neurons from the previous layer and preforms regression to a number of output neurons.
        Parameters:
            input (np.ndarray): Feature map of neurons from the preceeding layer.
        Returns:
            output (np.ndarray): A 3D matrix of shape n input features x output neurons
        """

        input = np.array(input).flatten()
        out = np.dot(input, self.w) + self.b

        # Record the incoming shape and data for backprop
        self.input = input
        self.out_preactivation = out

        out = self.act_function(out)
        self.output = out
        return out

    def backward(self, derivatives, learning_rate):
        """
        Backward pass through dense. Takes the derivative calculated in the next layer, calculates the error with respect to this layers weights, bais and input.
        It then updates the weights and biases in this layer based on the errors calculated * the models learning rate.
        Parameters:
            input (np.ndarray): Derivatives calculated in the following layer.
        Returns:
            output (np.ndarray): Derivatives with respect to this layers input.
        """

        d_error_d_out = derivatives

        d_error_d_input = np.dot(d_error_d_out, self.w.T)
        d_error_d_in = self.act_derivative_function(self.out_preactivation)
        d_input_d_w = np.reshape(self.input, (1, self.input.shape[0]))

        d_error_d_w = np.dot(d_input_d_w.T, d_error_d_out * d_error_d_in)
        d_error_d_b = np.mean(derivatives)


        # Update w and b
        self.w -= learning_rate * d_error_d_w
        self.b -= learning_rate * d_error_d_b

        # Gradient clipping
        if self.clip_value is not None:
            d_error_d_input[d_error_d_input < -self.clip_value] = -self.clip_value
            d_error_d_input[d_error_d_input > self.clip_value] = self.clip_value
            
        return np.reshape(d_error_d_input, self.input.shape)
