


from cProfile import label
from time import time
from typing import Tuple

import numpy as np
from model.cnn.layer.layer import Layer
from model.cnn.loss import cross_entropy_derivative, cross_entropy_label_loss
from model.model import Model
from model.scoring.metrics import Metrics
from persist.model_persister import ModelPersister
from util.console_utils import print_progress_bar


class SequentialModel(Model):
    """
    Defines the model structure for the neural network.
    Sequential refers to a succesion of Layers were the output of one layer becomes the input of the next.
    The list of Layers is then reversed for back propagation.
    """

    def __init__(self, name="") -> None:
        super().__init__(name)
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def summarize(self):
        print("Model Summary")
        print("-----------------------------")
        for layer in self.layers:
            print(layer.summarize())
            print("-----------------------------")

    def _forward_pass(self, image):
        for layer in self.layers:
            image = layer.forward(image)

        return image

    def _backward_pass(self, derivative, learning_rate):
        reveresed_layers = reversed(self.layers)
        for layer in reveresed_layers: 
            derivative = layer.backward(derivative, learning_rate)

    def compile(self):
        previous_layer = None
        for layer in self.layers:
            layer.compile(previous_layer)
            previous_layer = layer

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs, learning_rate=0.001, validation_data:Tuple=None, batch_size=None):
        # Persist some the parameters for model indentification
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create epoch history objects
        self.training_loss = []
        self.training_accuracy = []

        self.validation_loss = []
        self.validation_accuracy = []

        for epoch in range(1, epochs + 1):
            n_training_correct = 0
            epoch_training_loss = 0

            n_validation_correct = 0
            epoch_validation_loss = 0

            if batch_size is not None and batch_size > 0:
                # Mini batch gradient descent
                # Randomly sample data of size = batch size
                batch_indices = np.random.choice(np.arange(x_train.shape[0]), batch_size, replace=False)
                x_train_batch = x_train[batch_indices]
                y_train_batch = y_train[batch_indices]
            else:
                # Gradient descent
                x_train_batch = x_train
                y_train_batch = y_train


            for i in range(x_train_batch.shape[0]):
                print_progress_bar(i, x_train_batch.shape[0], "Epoch {0} progress: ".format(epoch), 10)
                input = x_train_batch[i]
                label = y_train_batch[i]

                # Feed forward
                out = self._forward_pass(input)

                if np.argmax(out) == label:
                    n_training_correct += 1
                
                # One hot encode the labels and set the positive class to 1.0
                y_hot = np.zeros(10, dtype=np.float32)
                y_hot[label] = 1.0
                
                # Compute loss for the target label pred for correct class
                loss = cross_entropy_label_loss(out[0][label])
                epoch_training_loss += loss
                

                # Calculate the inital derivatives
                # Using the error for each label obtained for the OHE correct label
                d_error_d_output = out[0] - y_hot
                self._backward_pass(d_error_d_output, learning_rate)
                

            # Calculate average loss and accuaracy for the epoch
            epoch_train_accuracy = n_training_correct / float(x_train_batch.shape[0])
            epoch_train_ave_loss = epoch_training_loss / float(x_train_batch.shape[0])

            self.training_accuracy.append(epoch_train_accuracy)
            self.training_loss.append(epoch_train_ave_loss)

            print("Epoch train ave. loss {}".format(epoch_train_ave_loss))
            print("Epoch train accuracy = {}".format(epoch_train_accuracy))

            if validation_data is not None:
                for i in range(validation_data[0].shape[0]):
                    input = validation_data[0][i]
                    label = validation_data[1][i]

                    # Feed forward
                    out = self._forward_pass(input)

                    if np.argmax(out) == label:
                        n_validation_correct += 1
                    
                    # One hot encode the labels and set the positive class to 1.0
                    y_hot = np.zeros(10, dtype=np.float32)
                    y_hot[label] = 1.0
                    
                    # Compute loss for the target label pred for correct class
                    loss = cross_entropy_label_loss(out[0][label])
                    epoch_validation_loss += loss
            
                epoch_validation_accuracy = n_validation_correct / float(validation_data[0].shape[0])
                epoch_validation_ave_loss = epoch_validation_loss / float(validation_data[0].shape[0])

                self.validation_accuracy.append(epoch_validation_accuracy)
                self.validation_loss.append(epoch_validation_ave_loss)

                print("Epoch validation ave. loss {}".format(epoch_validation_ave_loss))
                print("Epoch validation accuracy = {}".format(epoch_validation_accuracy))

        history = {}
        history['train_loss'] = np.asarray(self.training_loss)
        history['train_accuracy'] = np.asarray(self.training_accuracy)
        history['validation_loss'] = np.asarray(self.validation_loss)
        history['validation_accuracy'] = np.asarray(self.validation_accuracy)

        ModelPersister.persist(self, "nn\\final_softmax")

        return history

    def predict(self, x_test):
        y_pred = np.zeros(x_test.shape[0])

        for i in range(x_test.shape[0]):
            input = x_test[i]

            # Feed forward
            out = self._forward_pass(input)
            label = np.argmax(out)
            y_pred[i] = label

        return y_pred

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test).astype(np.int8)

        model_metrics = {
            'accuracy': Metrics.accuracy(y_test, y_pred),
            'precision': Metrics.precision(y_test, y_pred),
            'recall': Metrics.recall(y_test, y_pred),
            'f1': Metrics.f1(y_test, y_pred)
        }

        confusion_matrix = Metrics.confusion_matrix(
            y_test, y_pred, np.arange(10, dtype=np.int32))

        return model_metrics, confusion_matrix
