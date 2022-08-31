from typing import Tuple
import numpy as np

from model.cnn.activation import sigmoid
from model.cnn.loss import cross_entropy
from model.cnn.weight_types import random_uniform_weights, xavier_uniform_weights
from model.scoring.metrics import Metrics
from util.console_utils import print_progress_bar


class LogisticRegression():
    EARLY_STOP = 1e-3

    def __init__(self, learning_rate=0.01, num_iterations=1500, threshold=0.5,  weight_init_type='xavier'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.weight_init_type = weight_init_type

    def _init_weights_bias(self, x_train:np.ndarray):
        # Setting initial weights and biases

        # Determine the weight matrix from x_train demensions
        if x_train.ndim == 1:
            x_train_vector_dimension = 1
        else:
            x_train_vector_dimension = x_train.shape[1]

        if self.weight_init_type == 'xavier':
            self.w = xavier_uniform_weights((x_train_vector_dimension, 1), x_train_vector_dimension)
        else:
            self.w = random_uniform_weights((x_train_vector_dimension, 1))

        self.b = 0.0


    def _forward(self, x):
        x = np.array(x).reshape((x.shape[0], -1)).T
        z = np.dot(self.w.T, x) + self.b
        return sigmoid(z)

    def fit(self, x_train, y_train, validation_data:Tuple=None):
        self._init_weights_bias(x_train)

        # History objects
        training_loss = []
        training_accuracy = []

        validation_loss = []
        validation_accuracy = []

        for i in range(self.num_iterations):
            print_progress_bar(i, self.num_iterations, "LR Model Training", 10)
            z = self._forward(x_train)

            # Gradient decent
            # Element wise derivative
            d_error_d_w = (np.dot(x_train.T, ((z-y_train).T))) / x_train.shape[0]
            # Averaged difference for all training values
            d_error_d_b = np.mean(z-y_train)

            # Backward propagation adjust the weights and biases to find local minimum
            self.w = self.w - (self.learning_rate * d_error_d_w)
            self.b = self.b - (self.learning_rate * d_error_d_b)
            

            # Gather metrics on train & validation every 10 iterations to avoid large amounts of data
            train_loss = cross_entropy(y_train, z)
            average_train_loss = np.mean(train_loss)
            training_loss.append(average_train_loss)

            z_train = z
            z_train[z_train >= self.threshold] = 1
            z_train[z_train < 1] = 0
            train_accuracy = np.sum(y_train == z_train.flatten()) / x_train.shape[0]
            training_accuracy.append(train_accuracy)
            
            if abs(average_train_loss) <= LogisticRegression.EARLY_STOP:
                print("Stopping early at {0} iterations".format(i))
                break

            if i % 5 == 0:

                # Validation preformed every 10 iterations
                if validation_data is not None:
                    x_val = validation_data[0]
                    y_val = validation_data[1]

                    z_val = self._forward(x_val)

                    val_loss = cross_entropy(y_val, z_val)
                    average_val_loss = np.mean(val_loss)
                    validation_loss.append(average_val_loss)

                    z_val[z_val >= self.threshold] = 1
                    z_val[z_val < 1] = 0
                    val_accuracy = np.sum(y_val == z_val.flatten()) / x_val.shape[0]
                    validation_accuracy.append(val_accuracy)

        # Recording an returning history for visualising loss / accuracy curves
        history = {}
        history['train_loss'] = np.asarray(training_loss)
        history['train_accuracy'] = np.asarray(training_accuracy)
        history['validation_loss'] = np.asarray(validation_loss)
        history['validation_accuracy'] = np.asarray(validation_accuracy)

        return history

    def predict_prob(self, x_test) -> np.ndarray:
        return self._forward(x_test)

    def predict(self, x_test) -> np.ndarray:
        z = self.predict_prob(x_test)
        prediction_matrix = np.zeros(x_test.shape[0], dtype=np.int32)

        for i in range(z.shape[1]):
            if z[0, i] <= self.threshold:
                prediction_matrix[i] = 0
            else:
                prediction_matrix[i] = 1

        return prediction_matrix

    def score(self, x_test, y_test) -> Tuple[dict, np.ndarray]:
        y_pred = self.predict(x_test)
        y_prob = self.predict_prob(x_test).flatten()

        precision, recall, _ = Metrics.precision_recall_curve(y_test, y_prob)
        tpr, fpr, _ = Metrics.roc_curve(y_test, y_prob)
        model_metrics = {
            'accuracy': Metrics.accuracy(y_test, y_pred),
            'precision': Metrics.precision(y_test, y_pred),
            'recall': Metrics.recall(y_test, y_pred),
            'f1': Metrics.f1(y_test, y_pred),
            'precision_recall_curve': [precision, recall],
            'roc_curve': [tpr, fpr]
        }

        confusion_matrix = Metrics.confusion_matrix(
            y_test, y_pred, np.arange(2, dtype=np.int32))

        return model_metrics, confusion_matrix
