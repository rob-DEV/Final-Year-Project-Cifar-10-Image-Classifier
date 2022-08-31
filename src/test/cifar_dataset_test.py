import unittest

import numpy as np
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils


class CifarDatasetTests(unittest.TestCase):

    def test_cifar_load(self):
        x_train, y_train, x_test, y_test = CifarDataset.load()

        # assert sample count and label count are equal
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])

        # assert cifar image 32x32x3 shape in x_ data
        self.assertEqual(x_train.shape[1:4], (32, 32, 3))
        self.assertEqual(x_test.shape[1:4], (32, 32, 3))

    def test_map_labels_to_organic_inorganic(self):
        y_labels = np.arange(10)
        y_labels = CifarDatasetUtils.map_labels_to_organic_inorganic(y_labels)

        self.assertTrue(np.array_equal(
            y_labels, np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])))

    def test_map_labels_to_binary(self):
        y_labels = np.arange(10)
        y_labels = CifarDatasetUtils.map_labels_to_binary(y_labels, 2)

        self.assertTrue(np.array_equal(
            y_labels, np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])))


if __name__ == '__main__':
    unittest.main()
