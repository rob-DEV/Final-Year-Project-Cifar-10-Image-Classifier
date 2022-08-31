import numpy as np
from model.cnn.layer.layer import Layer


class Dropout(Layer):

    def __init__(self, name="", dropout_rate=0.1) -> None:
        super().__init__(name, activation=None)
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def compile(self, previous_layer):
        self.input_shape = previous_layer.output_shape
        self.output_shape = self.input_shape

    def forward(self, input: np.ndarray):
        self.dropout_mask = np.random.rand(*input.shape) * self.dropout_rate
        return input * self.dropout_mask

    def backward(self, derivative, learning_rate):
        return derivative *  self.dropout_mask
