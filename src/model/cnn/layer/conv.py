from typing import Tuple
from urllib.parse import non_hierarchical
import numpy as np
from image.convolution import convolve_2d_cnn_fast
from model.cnn.layer.layer import Layer
from model.cnn.weight_types import xavier_uniform_weights


class Conv(Layer):
    """
    Convolution layer, applies a series of filters to input images and aims to adjust these filters using back propagation to improve
    network accuracy.
    """

    def __init__(self, name="", activation="relu", input_shape:Tuple=None, filters=1, filter_shape=(1, 1)) -> None:
        """
        Initializes a convolutional layer with the standard name, activation along with the filters to be used for the convolutions.
            name (str): Mainly used for debugging and also printed in summarize.
            activation (str): Activation string to determine the activation function and its derivative (see activation.py)
            input_units (int): Number of inputs in this layer, corresponds to the flattened number of outputs in the previous layer.
            filters (int): Number of randomly initialized filters to convolve across the image, n filters will create an output of n x convolved_size output
            size (Tuple): X,Y size of the filters 3x3, 5x5 etc.
        """
        super().__init__(name, activation)
        self.input_shape = input_shape

        self.num_filters = filters
        self.filter_shape = filter_shape
        

    def compile(self, previous_layer: Layer):
        """
        Called during the compilation of a model. This method provides a way for the network to calculate the shapes for the input and output of the layer.
        """
        if previous_layer is None:
            # Ensure this layer (the first) has an input shape and determine the output shape
            if self.input_shape is None:
                raise Exception("No input shape has been provided for the first layer of the network!")

        if self.input_shape is None and previous_layer.output_shape is None:
            raise Exception("Unable to determine the input shape as the previous layer has no defined output shape")
            
        if self.input_shape is None:
            self.input_shape = previous_layer.output_shape

        self.filters = []
        for _ in range(self.num_filters):
            f = xavier_uniform_weights(self.filter_shape, np.prod(self.input_shape))
            self.filters.append(f)
        
       

        # Determine the output shape and set for compilation in the next layer
        output_feature_shape = int((self.input_shape[1] - self.filter_shape[1])) + 1
        self.output_shape = (self.num_filters, output_feature_shape, output_feature_shape)


    def forward(self, input):
        """
        Forward pass through convolution. Takes a feature map or image as input and applies a succsession of filters outputing them in a matrix of feature maps.
        Parameters:
            input (np.ndarray): Image or feature map in single or multi channel.
        Returns:
            output (np.ndarray): A stack of convolved images 1 for each filter.
        """
        # Store input for back prop
        self.forward_input = input

        out = np.zeros(self.output_shape)

        # Convolve for each filter
        for f_index in range(len(self.filters)):

            # Tile the Kernel aross the width / height of the image
            kernel = np.tile(self.filters[f_index], (input.shape[0], 1, 1))

            c = convolve_2d_cnn_fast(input, kernel)

            # Sum all channels to one feature map
            out[f_index] = c

        # Store output for back prop
        self.out_preactivation = out
        # Apply activation function
        out = self.act_function(out)
        self.output = out
        return out

    def backward(self, derivatives, learning_rate):
        """
        Backward pass of the layer. Attempts to calculate the errors with respect to the filters and the input.
        It then updates the filters based on the error values.
        Parameters:
            derivatives (np.ndarray): Derivatives from next layer in the network.
        Returns:
            output (np.ndarray): The derivaties with respect to layer input for use in the back propagation of the
            preceeding layer.
        """
        derivatives = derivatives.reshape(self.output_shape)
        derivatives = self.act_derivative_function(derivatives)

        # Derivative out with respect to the previous layer out (this layer in)
        d_error_d_input = np.zeros(self.forward_input.shape)
        d_error_d_filter = np.zeros(
            (self.num_filters, self.filter_shape[0], self.filter_shape[1]))

        for f_index in range(len(self.filters)):
            # Attempting vectorized convolution
            from numpy.lib.stride_tricks import sliding_window_view

            kernel = np.tile(self.filters[f_index],
                             (self.forward_input.shape[0], 1, 1))

            sliding_kernel_views = sliding_window_view(
                self.forward_input, kernel.shape)

            # Error with respect to the output / filters
            kernel = np.tile(derivatives[f_index],
                             (self.forward_input.shape[0], 1, 1))
            sliding_kernel_views = sliding_window_view(
                self.forward_input, kernel.shape)
            derivative_conv = sliding_kernel_views * kernel

            derivative_conv = derivative_conv.reshape(
                *derivative_conv.shape[:kernel.ndim], -1)
            derivative_conv = np.sum(
                derivative_conv, axis=kernel.ndim, keepdims=True)

            d_error_d_filter = derivative_conv.squeeze()

            # Error with respect to the input
            filters_kernel = np.tile(
                self.filters[f_index], (self.forward_input.shape[0], 1, 1))
            sliding_kernel_views = sliding_window_view(
                self.forward_input, filters_kernel.shape)
            filter_conv = sliding_kernel_views * filters_kernel

            filter_conv = filter_conv.reshape(
                *filter_conv.shape[:filters_kernel.ndim], -1)
            filter_conv = np.sum(
                filter_conv, axis=filters_kernel.ndim, keepdims=True)

            filter_conv = filter_conv.squeeze()

            d_error_d_input[:] += np.sum(filter_conv * derivatives)

            # Update the filter parameters
            self.filters[f_index] -= learning_rate * d_error_d_filter

        return d_error_d_input


    def backward_loop_version(self, derivatives, learning_rate):
        derivatives = derivatives.reshape(self.output_shape)
        derivatives = self.act_derivative_function(derivatives)

        # print("Max Pool Input Shape: {}".format(self.input.shape))
        # print("Max Pool Output Shape: {}".format(self.output.shape))
        # print("Derivative Shape: {}".format(derivatives.shape))

        # Unpack input shape
        _, input_x, input_y = self.forward_input.shape

        # Derivative out with respect to the previous layer out (this layer in)
        d_error_d_input = np.zeros(self.forward_input.shape)
        d_error_d_filter = np.zeros(
            (self.num_filters, self.filter_shape[0], self.filter_shape[1]))

        # Convolve for each filter
        for f_index in range(len(self.filters)):
            # Attempting to convolve across all channels at once and multiply
            # by the mask to stop excessive iteration
            x = 0
            out_x = 0
            while x + self.filter_shape[0] <= input_x:
                y = 0
                out_y = 0
                while y + self.filter_shape[1] <= input_y:
                    # Chunk of channel_count3x3
                    image_chunk = self.forward_input[:, x:x +
                                                     self.filter_shape[0], y:y + self.filter_shape[1]]

                    # Sum of the loss at the kernel (x,y) * image chunk
                    sum = np.sum(
                        derivatives[f_index, out_x, out_y] * image_chunk)
                    d_error_d_filter[f_index] += sum

                    # Error with repsect to the change in the filters
                    # d_error_d_input =
                    d_error_d_input[:, x:x+self.filter_shape[0], y:y+self.filter_shape[1]
                                    ] += derivatives[f_index, out_x, out_y] * self.filters[f_index]

                    y += 1
                    out_y += 1
                x += 1
                out_x += 1

        # Update the filter
        for f_index in range(len(self.filters)):
            self.filters[f_index] -= learning_rate * d_error_d_filter[f_index]

        return d_error_d_input
