import numpy as np


class CifarDatasetUtils:
    """
    This class provides utility functions for CIFAR-10 including,
    mapping classes to binary values and retrieving a samples classification
    as a string.
    """

    @staticmethod
    def map_labels_to_binary(y_labels, label_target, x_data=None):
        """
        Maps a list of labels to a binary vector were target_label is set to 1 and all other classifications become 0
        """
        # Create a copy of y_train
        y_labels = np.array(y_labels)

        # Set to a temp invalid label
        y_labels[y_labels != label_target] = -1

        # Target positive labels and set to 1
        y_labels[y_labels == label_target] = 1

        # Set all non 1 labels to 0
        y_labels[y_labels == -1] = 0

        # For cleaner assignment, ensures orginal data is not modified
        if x_data is None:
            return y_labels

        return np.array(x_data), y_labels

    def map_labels_to_organic_inorganic(y_labels, x_data=None):
        """
        Maps a list of labels to organic vs inorganic where organic is the positive class
        """
        # Airplane, automobile, ship, truck
        inorganic_classes = [0, 1, 8, 9]
        # Bird, cat, deer, dog, frog, horse
        # Organic_classes = [2,3,4,5,6,7]

        y_labels = np.array(y_labels)

        # There must be a numpy one liner for this
        for i in range(len(inorganic_classes)):
            inorganic_class = inorganic_classes[i]
            y_labels[y_labels == inorganic_class] = 0

        # Set non zero labels to 1
        y_labels[y_labels != 0] = 1

        if x_data is None:
            return y_labels

        return np.array(x_data), y_labels

    @staticmethod
    def classifications(index=None):
        """
        Utility function returning a list of the CIFAR-10 classifications text representations.
        """

        classifications = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"
            ]

        if index is not None:
            return classifications[index]
        else:
            return classifications

    @staticmethod
    def label_as_string(label):
        """
        Utility function for returning a classifiaction as text from an integer label.
        """
        classifications = CifarDatasetUtils.classifications()
        return classifications[label] if label >= 0 and label <= len(classifications) - 1 else "Invalid Classification"

    @staticmethod
    def normalize_data_min_max_scale(x):
        """
        Scales a dataset using min max scaling for dataset normalisation
        """
        x = np.array(x)
        maximum = np.max(x)
        minimum = np.min(x)
        normalized = (x - minimum) / (maximum - minimum)
        return normalized

    @staticmethod
    def split_train_validation_data(x_train, y_train, validation_set_size):
        if validation_set_size > 0:
            if validation_set_size >= x_train.shape[0]:
                raise Exception(
                    "Validation size must not exceed or equal the training size!")

            # Extract a validation sample set from x and update x
            validation_indexes = np.random.choice(
                np.arange(x_train.shape[0]), validation_set_size, replace=False)
            x_val = x_train[validation_indexes]
            y_val = y_train[validation_indexes]

            # Remove validation data from the training set
            x_train_mask = np.ones(x_train.shape[0], dtype=bool)
            x_train_mask[validation_indexes] = 0

            x_train = x_train[x_train_mask]
            y_train = y_train[x_train_mask]

        return x_train, y_train, x_val, y_val
