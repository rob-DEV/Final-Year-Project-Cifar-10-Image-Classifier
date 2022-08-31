import numpy as np

from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from feature_extraction.hog import HOG
from matplotlib import pyplot as plt
from model.logistic_regression.lr import LogisticRegression
from model.persist.lr_persistor import LogisticRegressionPersistor
from model.scoring.metrics import Metrics

def run_lr_organic_vs_inorganic():
    x_train, y_train, x_test, y_test = CifarDataset.load()

    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    hog = HOG()
    classified_hog_data = []

    with open('data\\feature_extraction\\classified_hog_data_chunk_8x8', 'rb') as file:
        classified_hog_data = np.load(file, allow_pickle=True)

    # Converted to standard format
    x_train = np.array(list(map(lambda x: x[0], classified_hog_data)))
    y_train = np.array(list(map(lambda x: x[1], classified_hog_data)))

    print("Calculating HOG vectors...")
    x_test = np.array(list(map(lambda x: hog.extract(x), x_test)))

    # Mapping to organic 1 vs inorganic 0
    x_train, y_train = CifarDatasetUtils.map_labels_to_organic_inorganic(
        y_train, x_train)
    x_test, y_test = CifarDatasetUtils.map_labels_to_organic_inorganic(
        y_test, x_test)

    model = LogisticRegression(learning_rate=0.01, num_iterations=600)

    model.fit(x_train, y_train)

    model_metrics, confusion_matrix = model.score(x_test, y_test)

    print(model_metrics)

    Metrics.plot_confusion_matrix(confusion_matrix)

    probs = model.predict_prob(x_test)
    tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs, 1)

    # Find optimal threshold
    max_arg = np.argmax(np.subtract(tpr, fpr))
    threshold = thresholds[max_arg]
    print("Optimal threshold: {0}".format(threshold))

    plt.plot(fpr, tpr, label="Organic vs Inorganic")

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()
    plt.title("HOG Organic vs Inorganic LR")
    plt.show()

def run_lr_organic_vs_inorganic_persisted_model():
    x_train, y_train, x_test, y_test = CifarDataset.load()

    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    hog = HOG()
    classified_hog_data = []

    with open('data\\feature_extraction\\classified_hog_data_chunk_8x8', 'rb') as file:
        classified_hog_data = np.load(file, allow_pickle=True)

    # Converted to standard format
    x_train = np.array(list(map(lambda x: x[0], classified_hog_data)))
    y_train = np.array(list(map(lambda x: x[1], classified_hog_data)))

    print("Calculating HOG vectors...")
    x_test = np.array(list(map(lambda x: hog.extract(x), x_test)))

    # Mapping to organic 1 vs inorganic 0
    x_train, y_train = CifarDatasetUtils.map_labels_to_organic_inorganic(
        y_train, x_train)
    x_test, y_test = CifarDatasetUtils.map_labels_to_organic_inorganic(
        y_test, x_test)

    model = LogisticRegressionPersistor.load("organic_vs_inorganic_hog_3000_0.01_0.5")

    confusion_matrix = model.score(x_test, y_test)

    Metrics.plot_confusion_matrix(confusion_matrix)

    probs = model.predict_prob(x_test)
    tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs, 1)
    plt.plot(fpr, tpr, label="Organic vs Inorganic")

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()
    plt.title("HOG Organic vs Inorganic LR")
    plt.show()

