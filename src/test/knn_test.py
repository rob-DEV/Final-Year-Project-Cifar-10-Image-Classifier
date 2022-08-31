import unittest

import numpy as np
from model.knn.k_nearest_neighbour import KNearestNeighbor


class KnnTests(unittest.TestCase):

    def setUp(self):
        self.knn = KNearestNeighbor()

        X_train = np.array([
            [543, 127, 28],
            [31, 765, 563],
            [616, 145, 224],
            [29, 15, 219]
        ])

        y_train = np.array([0, 1, 2, 3])

        self.knn.fit(X_train, y_train)

    def test_knn_has_accuracy_1_when_k_is_1_and_training_testing_equal(self):
        # Fitting training to training should be 100% accurate at k=1 as distances are all 0
        model_metrics, confusion_matrix = self.knn.score(
            self.knn.x_train, self.knn.y_train, k=1)
        self.assertEquals(model_metrics['accuracy'], 1.0)

    def test_knn_has_accuracy_less_than_1_when_k_greater_than_1_and_training_testing_equal(self):
        # Fitting training to training should be less than 100% accurate at k=1 as distances are all > 0
        model_metrics, confusion_matrix = self.knn.score(
            self.knn.x_train, self.knn.y_train, k=2)
        self.assertLess(model_metrics['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
