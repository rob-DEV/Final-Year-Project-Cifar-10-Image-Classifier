

import pickle
import numpy as np

from util.console_utils import print_progress_bar


class CifarFeaturePersister:
    _BASE_PATH = "data\\feature_extraction\\cifar_10_"

    @staticmethod
    def persist(x: np.ndarray, y: np.ndarray, extraction_class, file_name):

        xy_dict = {}

        features = []

        for i in range(x.shape[0]):
            print_progress_bar(
                i, x.shape[0], "Cifar 10 Feature extraction progress: ".format(i), 10)
            features.append(extraction_class.extract(x[i]))

        xy_dict['x'] = np.asarray(features)
        xy_dict['y'] = y

        with open(CifarFeaturePersister._BASE_PATH + file_name, 'wb') as file:
            pickle.dump(xy_dict, file)

    @staticmethod
    def load(name, sample_size=None):
        with open(CifarFeaturePersister._BASE_PATH + name, 'rb') as file:
            xy_dict = pickle.load(file)
            x = xy_dict['x']
            y = xy_dict['y']

            if sample_size is not None:
                random_indexes = np.random.choice(np.arange(x.shape[0]), sample_size, replace=False)
                x = x[random_indexes]
                y = y[random_indexes]

            return x, y
