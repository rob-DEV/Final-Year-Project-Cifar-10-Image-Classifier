import numpy as np
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from feature_extraction.hog import HOG
from matplotlib import pyplot as plt
from model.logistic_regression.lr import LogisticRegression
from model.persist.lr_persistor import LogisticRegressionPersistor
from model.scoring.metrics import Metrics


def run_lr_one_vs_all():
    x_train, y_train, x_test, y_test = CifarDataset.load()

    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    hog = HOG()
    classified_hog_data = []

    with open('data\\feature_extraction\\classified_hog_data_chunk_8x8', 'rb') as file:
        classified_hog_data = np.load(file, allow_pickle=True)

    # converted to standard format
    x_train = np.array(list(map(lambda x: x[0], classified_hog_data)))
    y_train = np.array(list(map(lambda x: x[1], classified_hog_data)))

    print("Calculating HOG vectors...")
    x_test = np.array(list(map(lambda x: hog.extract(x), x_test)))

    for i in range(10):
        x_train_lr, y_train_lr = CifarDatasetUtils.map_labels_to_binary(
            y_train, i, x_train)
        x_test_lr, y_test_lr = CifarDatasetUtils.map_labels_to_binary(
            y_test, i, x_test)

        # model = LogisticRegression(learning_rate=0.005, num_iterations=5000)
        # model.fit(x_train_lr, y_train_lr)
        # persist(model,"one_vs_all_5000_0.005/{0}".format(i))

        model = LogisticRegressionPersistor.load(
            "one_vs_all_10000_0.0005/{0}".format(i))
        probs = model.predict_prob(x_test)
        tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs, i)

        max_arg = np.argmax(np.subtract(tpr, fpr))
        threshold = thresholds[max_arg]
        print("Optimal threshold for {0}: {1}".format(
            CifarDatasetUtils.label_as_string(i), threshold))

        plt.plot(fpr, tpr, label=CifarDatasetUtils.label_as_string(i))

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()
    plt.show()


def run_lr_one_vs_all():
    x_train, y_train, x_test, y_test = CifarDataset.load()

    x_test = x_test[0:1000]
    y_test = y_test[0:1000]

    hog = HOG()
    classified_hog_data = []

    with open('data\\feature_extraction\\classified_hog_data_chunk_8x8', 'rb') as file:
        classified_hog_data = np.load(file, allow_pickle=True)

    # converted to standard format
    x_train = np.array(list(map(lambda x: x[0], classified_hog_data)))
    y_train = np.array(list(map(lambda x: x[1], classified_hog_data)))

    print("Calculating HOG vectors...")
    x_test = np.array(list(map(lambda x: hog.extract(x), x_test)))

    y_pred_list = []
    for i in range(10):
        x_train_lr, y_train_lr = CifarDatasetUtils.map_labels_to_binary(
            y_train, i, x_train)
        x_test_lr, y_test_lr = CifarDatasetUtils.map_labels_to_binary(
            y_test, i, x_test)

        # model = LogisticRegression(learning_rate=0.005, num_iterations=5000)
        # model.fit(x_train_lr, y_train_lr)
        # persist(model,"one_vs_all_5000_0.005/{0}".format(i))

        model = LogisticRegressionPersistor.load("one_vs_all_10000_0.0005/{0}".format(i))
        probs = model.predict_prob(x_test)
        tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs, i)

        max_arg = np.argmax(np.subtract(tpr, fpr))
        threshold = thresholds[max_arg]
        print("Optimal threshold for {0}: {1}".format(
            CifarDatasetUtils.label_as_string(i), threshold))

        plt.plot(fpr, tpr, label=CifarDatasetUtils.label_as_string(i))

    plt.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')
    plt.legend()
    plt.show()
