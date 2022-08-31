import matplotlib.pyplot as plt
import numpy as np

from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from feature_extraction.canny_edge_extraction import CannyEdgeExtraction
from feature_extraction.edge_extraction import SobelEdgeExtraction
from feature_extraction.grayscale import Grayscale
from feature_extraction.histogram import ColorHistogram
from feature_extraction.hog import HOG
from image.image_ops import ImageOps
from model.cnn.layer.conv import Conv
from model.cnn.layer.dense import Dense
from model.cnn.layer.dropout import Dropout
from model.cnn.layer.max_pooling import MaxPooling
from model.cnn.sequential_model import SequentialModel
from model.knn.k_nearest_neighbour import KNearestNeighbor
from model.logistic_regression.lr import LogisticRegression
from model.scoring.metrics import Metrics
from persist.cifar_feature_persister import CifarFeaturePersister
from persist.model_persister import ModelPersister


def main():
    # Load CIFAR-10 dataset
    x_train, y_train, x_test, y_test = CifarDataset.load()
    
    # Load a particular feature (HOG blocksize=2x2)
    x_train, y_train = CifarFeaturePersister.load('hog_2x2_train')
    x_test, y_test = CifarFeaturePersister.load('hog_2x2_test')

    # Create and score KNN classifier using a small sample of test images
    knn = KNearestNeighbor()
    knn.fit(x_train, y_train)
    metrics, confusion_matrix = knn.score(x_test[0:500], y_test[0:500], k = 3)

    print("Accuracy: {}".format(metrics['accuracy']))
    print("F1: {}".format(metrics['f1']))

    # Create a organic vs inorganic classifier

    # Map labels to organic or inorganic
    x_train_log, y_train_log = CifarDatasetUtils.map_labels_to_organic_inorganic(y_train[0:2000], x_train[0:2000])
    x_test_log, y_test_log = CifarDatasetUtils.map_labels_to_organic_inorganic(y_test[0:2000], x_test[0:2000])

    organic_classifier = LogisticRegression(learning_rate=0.01, num_iterations=5000, weight_init_type='xavier')
    history = organic_classifier.fit(x_train_log, y_train_log)

    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['train_accuracy'], label='Training accuracy')

    # Test and plot metrics
    metrics, confusion_matrix = organic_classifier.score(x_test_log, y_test_log)

    Metrics.plot_confusion_matrix(confusion_matrix)


    # Create a softmax regression model MLP
    dense_model = SequentialModel('My dense network')
    dense_model.add(Dense('dense_1', activation='relu', input_shape=2025, output_units=1024))
    dense_model.add(Dense('dense_2', activation='relu', output_units=256))
    dense_model.add(Dense('output_dense', activation='softmax', output_units=10))

    dense_model.compile()

    history = dense_model.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.01, validation_data=(x_train[1001:2000],y_train[1001:2000]))

    metrics, confusion_matrix = dense_model.score(x_test[0:1000], y_test[0:1000])

    Metrics.plot_confusion_matrix(confusion_matrix)
    
    plt.show()

    # Create a softmax regression model MLP with CNN
    x_train, y_train, x_test, y_test = CifarDataset.load()

    # Reshape each sample to Nx3x32x32
    x_train = np.array(x_train).transpose((0, 3, 1, 2)).astype(np.float64)
    x_test = np.array(x_test).transpose((0, 3, 1, 2)).astype(np.float64)


    cnn_model = SequentialModel('My dense network')
    cnn_model.add(Conv('conv', activation='relu', input_shape=(3,32,32), filters=32, filter_shape=(3,3)))
    cnn_model.add(MaxPooling('max_pooling', pooling_shape=(2,2)))
    cnn_model.add(Conv('conv', activation='relu', input_shape=(3,32,32), filters=32, filter_shape=(3,3)))
    cnn_model.add(MaxPooling('max_pooling', pooling_shape=(2,2)))
    cnn_model.add(Dense('dense_1', activation='relu', output_units=1024))
    cnn_model.add(Dense('dense_2', activation='relu', output_units=256))
    cnn_model.add(Dense('output_dense', activation='softmax', output_units=10))

    cnn_model.compile()

    history = cnn_model.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.01, validation_data=(x_train[1001:2000],y_train[1001:2000]))

    metrics, confusion_matrix = cnn_model.score(x_test[0:1000], y_test[0:1000])

    Metrics.plot_confusion_matrix(confusion_matrix)
    
    plt.show()

    # Create a CNN model
    x_train, y_train, x_test, y_test = CifarDataset.load()

    # Reshape each sample to Nx3x32x32
    x_train = np.array(x_train).transpose((0, 3, 1, 2)).astype(np.float64)
    x_test = np.array(x_test).transpose((0, 3, 1, 2)).astype(np.float64)


    cnn_model = SequentialModel('My dense network')
    cnn_model.add(Conv("convolution_1", activation='relu', input_shape=(3,32,32), filters=32, filter_shape=(3, 3)))
    cnn_model.add(Conv("convolution_2", activation='relu', filters=32, filter_shape=(3, 3)))
    cnn_model.add(MaxPooling("max_pool_1", pooling_shape=(2, 2), stride=2))
    cnn_model.add(Conv("convolution_3", activation='relu', filters=64, filter_shape=(3, 3)))
    cnn_model.add(Conv("convolution_4", activation='relu', filters=64, filter_shape=(3, 3)))
    cnn_model.add(MaxPooling("max_pool_2", pooling_shape=(2, 2), stride=2))
    cnn_model.add(Dense("dense_1", activation='softmax', output_units=10))


    cnn_model.compile()

    history = cnn_model.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.01, validation_data=(x_train[1001:2000],y_train[1001:2000]))

    metrics, confusion_matrix = cnn_model.score(x_test[0:1000], y_test[0:1000])

    Metrics.plot_confusion_matrix(confusion_matrix)
    
    plt.show()


if __name__ == "__main__":
    main()
