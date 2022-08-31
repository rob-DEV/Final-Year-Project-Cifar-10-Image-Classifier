import numpy as np
import matplotlib.pyplot as plt
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from feature_extraction.hog import HOG
from model.knn.k_nearest_neighbour import KNearestNeighbor
from model.scoring.metrics import Metrics

def run_knn_one_vs_all():
    x_train, y_train, x_test, y_test = CifarDataset.load()
   
    x_train = x_train[0:50000]
    y_train = y_train[0:50000]
    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    knn = KNearestNeighbor()
    knn.fit(np.reshape(x_train, (x_train.shape[0], -1)), y_train)

    knn.score(np.reshape(x_test, (x_test.shape[0], -1)), y_test, k=5)

    probs = knn.predict_prob(np.reshape(x_test, (x_test.shape[0], -1)), k=5)

    print("Calculating ROC curve...")
    for i in range(10):
        tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs[:, i], i)
        plt.plot(fpr, tpr, label=CifarDatasetUtils.label_as_string(i))

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()

    plt.show()

def run_knn_hog_one_vs_all():
    x_train, y_train, x_test, y_test = CifarDataset.load()
    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    hog = HOG()
    classified_hog_data = []

    with open('data\\feature_extraction\\classified_hog_data_chunk_8x8', 'rb') as file:
        classified_hog_data = np.load(file, allow_pickle=True)

    x_train = np.array(list(map(lambda x: x[0], classified_hog_data)))
    y_train = np.array(list(map(lambda x: x[1], classified_hog_data)))
    
    print("Calculating HOG vectors for KNN...")
    x_test = np.array(list(map(lambda x: hog.extract(x), x_test)))

    knn = KNearestNeighbor()
    knn.fit(x_train, y_train)

    knn.score(x_test, y_test, k=5)

    probs = knn.predict_prob(np.reshape(x_test, (x_test.shape[0], -1)), k=5)

    print("Calculating ROC curve...")
    for i in range(10):
        tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs[:, i], i)
        plt.plot(fpr, tpr, label=CifarDatasetUtils.label_as_string(i))

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()

    plt.show()