from typing import Tuple
import numpy as np
from maths.maths import vectorized_euclidean_distance

from model.scoring.metrics import Metrics


class KNearestNeighbor():
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        # Flatten to a long vector
        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        self.x_train = x_train
        self.y_train = y_train

    def predict_k_closest_classes(self, x_test, k):
        x_test = np.reshape(x_test, (x_test.shape[0], -1))

        distances = vectorized_euclidean_distance(self.x_train, x_test)
        y_k_closest_classes = np.zeros((distances.shape[0], k), dtype=np.int32)

        for i in range(distances.shape[0]):
            sorted_distance_indices = np.argsort(distances[i])
            y_k_closest_classes[i] = self.y_train[sorted_distance_indices[0:k]]
        
        return y_k_closest_classes

    def predict(self, x_test, k):
        y_k_closest_classes = self.predict_k_closest_classes(x_test, k)
        y_pred = np.zeros(x_test.shape[0], dtype=np.int32)

        # Find mode of the classes and store prediction for y_test[i]
        for i in range(y_pred.shape[0]):
            y_pred[i] = np.argmax(np.bincount(y_k_closest_classes[i]))

        return y_pred

    def predict_prob(self, x_test, k):
        y_k_closest_classes = self.predict_k_closest_classes(x_test, k)

        prob = np.zeros((x_test.shape[0], 10))

        # Calculate the probability for each of the k closest classes
        for i in range(x_test.shape[0]):
            # Binning of the class occurencies in k closest classes
            prob[i] = np.bincount(y_k_closest_classes[i], minlength=10)
            # Ratio the probabilities so sum is 1.0
            prob[i] = prob[i] / np.sum(prob[i])
        
        return prob

    def score(self, x_test, y_test, k=1) -> Tuple[dict, np.ndarray]:
        x_test = np.reshape(x_test, (x_test.shape[0], -1))

        print("Running KNN at k={0}...".format(k))
        y_pred = self.predict(x_test, k=k)

        model_metrics = {
            'accuracy': Metrics.accuracy(y_test, y_pred),
            'precision': Metrics.precision(y_test, y_pred),
            'recall': Metrics.recall(y_test, y_pred),
            'f1': Metrics.f1(y_test, y_pred),
        }

        confusion_matrix = Metrics.confusion_matrix(
            y_test, y_pred, np.arange(10, dtype=np.int32))

        return model_metrics, confusion_matrix

