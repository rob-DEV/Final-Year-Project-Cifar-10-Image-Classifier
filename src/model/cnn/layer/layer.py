from model.cnn.activation import activation_derivative_function, activation_function


class Layer:
    """
    The base class for a layer in the CNN.
    Every other layer derives from this class and this class should not be instantiated directly.
    """

    def __init__(self, name, activation) -> None:
        """
        Initialises a layer with a name and determines it's activation function and activation derivative based the string provided. 
        Parameters:
            name (str): Mainly used for debugging and also printed in summarize.
            activation (str): Activation string to determine the activation function and its derivative (see activation.py)
        """
        self.name = name
        # Grab the appropriate activation function pointer by name
        self.activation = activation
        if self.activation != None:
            # Resolve the activation function and it's derivative function
            self.act_function = activation_function(activation)
            self.act_derivative_function = activation_derivative_function(
                activation)
        else:
            self.act_function = None
            self.act_derivative_function = None

    def compile(self, previous_layer):
        raise Exception(
            "Layer compile must not be called from the base class!")

    def summarize(self):
        """
        Prints layer information when setting up the network and when debugging.
        """
        return "Name: {0}\n\tType: {1}\n\tActivation: {2}\n\tFunc Pointer: {3}".format(self.name, type(self).__name__, self.activation, self.act_function)

    def forward(self, input = None):
        raise Exception(
            "Layer forward must not be called from the base class!")

    def backward(self, derivatives, learning_rate):
        raise Exception(
            "Layer backward must not be called from the base class!")
