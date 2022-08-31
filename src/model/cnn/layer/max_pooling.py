
from time import time
from typing import Tuple
from matplotlib.pyplot import axis
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from model.cnn.layer.layer import Layer


class MaxPooling(Layer):
    """
    Max pooling layer, takes a window shape of X by Y and convolves across the feature map and creates an output containing the max value in each window.
    """

    def __init__(self, name="", activation=None, pooling_shape=(2, 2), stride=2, input_shape:Tuple=None) -> None:
        """
        Initializes a max pooling layer with the standard name, activation along with the input units from the previous layer and the desired number of outputs.
            name (str): Mainly used for debugging and also printed in summarize.
            activation (str): No activation function will be applied in Max Pooling.
            size (Tuple): Window shape for pooling.
            stride (int): Step size in x and y which the pooling window should move after each pooling of the image 
        """
        super().__init__(name, activation)

        self.pooling_shape = pooling_shape
        self.stride = stride
        self.input_shape = input_shape

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

        # Determine the output shape and set for compilation in the next layer
        output_feature_shape = int((self.input_shape[1] - self.pooling_shape[1]) / self.stride) + 1
        self.output_shape = (self.input_shape[0], output_feature_shape, output_feature_shape)

    def forward(self, image: np.ndarray):
        """
        Forward pass through max pooling. Each feature map is broken into pooling windows at the appropriate stride, the max value becomes the output for that region.
        Parameters:
            input (np.ndarray): Image or feature map in single or multi channel.
        Returns:
            output (np.ndarray): Stack of max pooled maps
        """
        # Store input for back prop
        self.forward_input = image

        input_size = image.shape
        out = np.zeros(self.output_shape)

        for f_index in range(input_size[0]):
            input_as_strided_blocks = sliding_window_view(
                image[f_index, :, :], self.pooling_shape)[::self.stride, ::self.stride]
            input_as_strided_blocks = input_as_strided_blocks.reshape(
                *input_as_strided_blocks.shape[:len(self.pooling_shape)], -1)
            pooled_values = np.max(input_as_strided_blocks, axis=len(
                self.pooling_shape), keepdims=True)
            pooled_values = np.squeeze(pooled_values)

            out[f_index][0:pooled_values.shape[0], 0:pooled_values.shape[0]] = pooled_values

        return out

    def backward(self, derivatives, learning_rate):
        """
        Backward pass through max pooling. 
        Parameters:
            derivative (np.ndarray): Derivatives calculated in the next layer.
        Returns:
            output (np.ndarray): Stack of feature maps of the forward input.shape where each derivative is mapped to the max positions of the for input and the rest is 0
        """
        # Reshape derviatives to output shape
        derivatives = np.reshape(derivatives, self.output_shape)

        # Unpack input shape
        feature_maps, _, _ = self.forward_input.shape
        d_error_d_input = np.zeros(self.forward_input.shape)

        flattened_feature_maps_cache = self.forward_input.reshape(
            feature_maps, -1)

        for f_index in range(feature_maps):
            input_as_strided_blocks = sliding_window_view(
                self.forward_input[f_index, :, :], self.pooling_shape)[::self.stride, ::self.stride]
            input_as_strided_blocks = input_as_strided_blocks.reshape(
                *input_as_strided_blocks.shape[:len(self.pooling_shape)], -1)

            # Get the max values from the previous layer input
            pooled_values = np.max(input_as_strided_blocks, axis=len(
                self.pooling_shape)).flatten()

            # Repeat the derivative matrix by the stride for indexing with d_error_d_input
            unstrided_derivatives = np.zeros(self.forward_input[f_index].shape)
            tmp = np.repeat(derivatives[f_index], 2, axis=0).repeat(2, axis=1)

            unstrided_derivatives[0:tmp.shape[0], 0:tmp.shape[1]] = tmp

            # Flatten input
            filter_feature_map = flattened_feature_maps_cache[f_index]

            indexes = np.nonzero(
                pooled_values[:, None] == filter_feature_map)[1]

            d_filter_map = np.zeros(filter_feature_map.shape)
            MAX_MARKER = 9999
            d_filter_map[indexes] = MAX_MARKER

            # Merge the arrays based on max indexes
            d_filter_map[d_filter_map == MAX_MARKER] = unstrided_derivatives.flatten()[
                d_filter_map == MAX_MARKER]
            d_filter_map = d_filter_map.reshape(
                self.forward_input[f_index].shape)

            # Populate the derivaitve map at f_index
            d_error_d_input[f_index] = d_filter_map

            # plt.subplot(1,2,1)
            # plt.imshow(self.forward_input[f_index])
            # plt.subplot(1,2,2)
            # plt.imshow(filter_derivative_map)
            # plt.show()

        return d_error_d_input



