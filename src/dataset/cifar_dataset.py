from os import getcwd, listdir
from os.path import isfile, join

import numpy as np

from dataset.dataset_constants import DatasetConstants
class CifarDataset:
    """
    This class handles loading of the CIFAR-10 dataset in RGB format it will also cache the dataset to avoid excessive loading
    """

    # Cache of the Dataset
    _dataset = {}



    @staticmethod
    def _load_batches(batch_paths):
        batches = []
        for file_path in batch_paths:
            import pickle
            with open(file_path, 'rb') as fo:
                batches.append(pickle.load(fo, encoding='bytes'))

        batches = sorted(batches, key=lambda d: d[b'batch_label'])
        return np.array(batches)

    @staticmethod
    def _process_batch(batches: np.ndarray):
        data = []
        labels = []

        for batch in batches:
            X = batch[b'data']
            Y = batch[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(
                0, 2, 3, 1).astype(np.double)

            # Normalizing colour data
            X = X / 255.0

            Y = np.array(Y)

            data.append(X)
            labels.append(Y)

        data = np.concatenate(data)
        labels = np.concatenate(labels)
        del X, Y

        return data, labels

    @staticmethod
    def load():
        """
        Loads the CIFAR-10 training and test dataset normalizing the RGB pixel values
        """

        if len(CifarDataset._dataset) == 0:
            path_to_cifar = DatasetConstants.CIFAR_DATASET_PATH
            cifar = list(map(lambda file: join(
                getcwd(), path_to_cifar, file), listdir(path_to_cifar)))

            cifar_data = [x for x in cifar if "data_batch" in x and isfile(x)]
            cifar_test = [x for x in cifar if "test_batch" in x and isfile(x)]

            training_batches = CifarDataset._load_batches(cifar_data)
            test_batches = CifarDataset._load_batches(cifar_test)

            # Extract images and labels each to one large array
            x_train, y_train = CifarDataset._process_batch(training_batches)
            x_test, y_test = CifarDataset._process_batch(test_batches)

            CifarDataset._dataset['x_train'] = x_train
            CifarDataset._dataset['y_train'] = y_train
            CifarDataset._dataset['x_test'] = x_test
            CifarDataset._dataset['y_test'] = y_test

        return CifarDataset._dataset.values()

    @staticmethod
    def take_sample_of_cifar_train(sample_size):

        if len(CifarDataset._dataset) == 0:
            CifarDataset.load()

        x_train = CifarDataset._dataset['x_train']
        y_train = CifarDataset._dataset['y_train']
        random_indexes = np.random.choice(np.arange(x_train.shape[0]), sample_size, replace=False)

        x = x_train[random_indexes]
        y = y_train[random_indexes]

        return x, y

    @staticmethod
    def take_sample_of_cifar_test(sample_size):

        if len(CifarDataset._dataset) == 0:
            CifarDataset.load()

        x_train = CifarDataset._dataset['x_test']
        y_train = CifarDataset._dataset['y_test']
        random_indexes = np.random.choice(np.arange(x_train.shape[0]), sample_size, replace=False)

        x = x_train[random_indexes]
        y = y_train[random_indexes]

        return x, y