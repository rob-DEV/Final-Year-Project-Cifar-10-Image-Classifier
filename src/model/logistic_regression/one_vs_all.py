from typing import Tuple
import numpy as np
from dataset.cifar_dataset_utils import CifarDatasetUtils
from model.logistic_regression.lr import LogisticRegression
from model.model import Model
from model.scoring.metrics import Metrics


class Cifar10OneVsAll(Model):
    def __init__(self, learning_rate=0.01, iterations=1500):
        super().__init__("Cifar10OneVsAll")
        # Create a LR model for each of the cifar classes
        self.classifiers = []

        # Create a classifier for each label 0-9
        for _ in range(len(CifarDatasetUtils.classifications())):
            self.classifiers.append(LogisticRegression(learning_rate=learning_rate, num_iterations=iterations))

    def fit(self, x_train, y_train, validation_data:Tuple=None):
        histories = []
        for label in range(len(self.classifiers)):
            # Map CIFAR labels to 0 or 1 for the target class i
            y_train_mapped = CifarDatasetUtils.map_labels_to_binary(y_train, label)


            # Train the model
            print("Training model for class={}".format(label))
            if validation_data is not None:
                validation_data_mapped = CifarDatasetUtils.map_labels_to_binary(validation_data[1], label, validation_data[0])
                history = self.classifiers[label].fit(x_train, y_train_mapped, validation_data=validation_data_mapped)
            history = self.classifiers[label].fit(x_train, y_train_mapped, validation_data=None)

            histories.append(history)

        return histories

    def predict(self, x_test):
        # One vs all
        # Setup a matrix of n x 10 and argmax the highest probability as the prediction
        prediction_prob_matrix = np.zeros((x_test.shape[0], 10), dtype=np.float32)
        for label in range(len(self.classifiers)):
            probs = self.classifiers[label].predict_prob(x_test)
            probs = probs.flatten()
            prediction_prob_matrix[:, label] = probs

        # Max probability on axis=1 (2nd dimension) taken as the classification
        class_predictions = np.argmax(prediction_prob_matrix, axis=1)
        return class_predictions

    def score(self, x_test, y_test):
        classifier_metrics = []
        for label in range(len(self.classifiers)):
            # Map CIFAR labels to 0 or 1 for the target class i
            y_mapped = CifarDatasetUtils.map_labels_to_binary(y_test, label)

            # Train the model
            model_metrics, confusion_matrix = self.classifiers[label].score(x_test, y_mapped)

            classifier_metrics.append(model_metrics)

        y_pred = self.predict(x_test)
        confusion_matrix = Metrics.confusion_matrix(y_test, y_pred, np.arange(10, dtype=np.int32))

        metrics = {}
        metrics['precision'] = list(map(lambda x: x['precision'], classifier_metrics))
        metrics['recall'] = list(map(lambda x: x['recall'], classifier_metrics))
        metrics['f1'] = list(map(lambda x: x['f1'], classifier_metrics))
        metrics['accuracy'] = Metrics.accuracy(y_test, y_pred)
        metrics['precision_recall_curves'] = list(map(lambda x: x['precision_recall_curve'], classifier_metrics))
        metrics['roc_curves'] = list(map(lambda x: x['roc_curve'], classifier_metrics))

        return metrics, confusion_matrix