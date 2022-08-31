import numpy as np
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from feature_extraction.grayscale import Grayscale
from model.knn.k_nearest_neighbour import KNearestNeighbor
from model.scoring.metrics import Metrics
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGridLayout, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)
from persist.cifar_feature_persister import CifarFeaturePersister


class KNearestNeighbourView(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        parameter_layout = self.setup_parameter_layout()
        evaluation_layout = self.setup_evaluation_layout()

        layout.addLayout(parameter_layout, stretch=0)
        layout.addLayout(evaluation_layout, stretch=4)

        
        # Run the model initially to populate graphs
        self.evaluate_model()
        

    def setup_parameter_layout(self):
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addSpacing(10)
        self.parameter_layout.addWidget(QLabel('KNN Parameters'))

        parameter_form_layout = QFormLayout()

        self.knn_combo_box = QComboBox()
        self.knn_combo_box.addItems([
            'HOG',
            'Grayscale',
            'RGB',
            'Histogram',
            'Sobel',
            'Canny'
        ])

        self.knn_train_size_spin_box = QSpinBox()
        self.knn_train_size_spin_box.setMinimum(1)
        self.knn_train_size_spin_box.setMaximum(50000)
        self.knn_train_size_spin_box.setValue(100)

        self.knn_test_size_spin_box = QSpinBox()
        self.knn_test_size_spin_box.setMinimum(1)
        self.knn_test_size_spin_box.setMaximum(10000)
        self.knn_test_size_spin_box.setValue(10)

        self.knn_k_neighbours = QSpinBox()
        self.knn_k_neighbours.setMinimum(1)
        self.knn_k_neighbours.setMaximum(10)
        self.knn_k_neighbours.setValue(5)

        parameter_form_layout.addRow(
            'Comparison Feature: ', self.knn_combo_box)
        parameter_form_layout.addRow(
            'Train Data size: ', self.knn_train_size_spin_box)
        parameter_form_layout.addRow(
            'Test Data size: ', self.knn_test_size_spin_box)
        parameter_form_layout.addRow('K: ', self.knn_k_neighbours)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.evaluate_model)
        parameter_form_layout.addRow(run_button)

        self.parameter_layout.addLayout(parameter_form_layout)
        self.parameter_layout.addStretch()

        return self.parameter_layout

    def setup_evaluation_layout(self):
        self.evaluation_layout = QVBoxLayout()
        self.evaluation_layout.addSpacing(10)
        self.evaluation_layout.addWidget(QLabel('Model Evaluation'))

        self.evaluation_graph_layout = QGridLayout()
        self.evaluation_graph_layout.setRowStretch(0,2)
        self.evaluation_graph_layout.setRowStretch(1,1)

        self.tl_figure = plt.figure()
        self.tr_figure = plt.figure()
        self.bl_figure = plt.figure()

        self.tl_canvas = GraphCanvas(self.tl_figure)
        self.tr_canvas = GraphCanvas(self.tr_figure)
        self.bl_canvas = GraphCanvas(self.bl_figure)


        self.metrics_layout = QVBoxLayout()

        self.metrics_header = QLabel()
        self.metrics_accuracy = QLabel()
        self.metrics_precision = QLabel()
        self.metrics_recall = QLabel()
        self.metrics_f1 = QLabel()

        self.metrics_layout.addWidget(self.metrics_header)
        self.metrics_layout.addWidget(self.metrics_accuracy)
        self.metrics_layout.addWidget(self.metrics_precision)
        self.metrics_layout.addWidget(self.metrics_recall)
        self.metrics_layout.addWidget(self.metrics_f1)

        self.evaluation_graph_layout.addWidget(self.tl_canvas, 0, 0)
        self.evaluation_graph_layout.addWidget(self.tr_canvas, 1, 0)
        self.evaluation_graph_layout.addWidget(self.bl_canvas, 0, 1)
        self.evaluation_graph_layout.addLayout(self.metrics_layout, 1, 1)

        self.evaluation_results_layout = QHBoxLayout()

        self.evaluation_layout.addLayout(self.evaluation_graph_layout)
        self.evaluation_layout.addLayout(self.evaluation_results_layout)

        self.evaluation_layout.addStretch()

        return self.evaluation_layout

    def evaluate_model(self):
        # Gathers the data specified i.e. K smaple size and test images and runs KNN for evaluation.
        knn_feature = self.knn_combo_box.currentText()
        print("Running KNN Model on CIFAR using {}...".format(knn_feature))

        train_size = self.knn_train_size_spin_box.value()
        test_size = self.knn_test_size_spin_box.value()
        k = self.knn_k_neighbours.value()

        feature_to_extract = self.knn_combo_box.currentText()
        print(feature_to_extract)

        if feature_to_extract == "HOG":
            x_train, y_train = CifarFeaturePersister.load('hog_2x2_train', sample_size=train_size)
            x_test, y_test = CifarFeaturePersister.load('hog_2x2_test',  sample_size=test_size)
        elif feature_to_extract == "Grayscale":
            x_train, y_train = CifarFeaturePersister.load('grayscale_train', sample_size=train_size)
            x_test, y_test = CifarFeaturePersister.load('grayscale_test', sample_size=test_size)
        elif feature_to_extract == "Histogram":
            x_train, y_train = CifarFeaturePersister.load('color_histogram_train', sample_size=train_size)
            x_test, y_test = CifarFeaturePersister.load('color_histogram_test', sample_size=test_size)
        elif feature_to_extract == "Sobel":
            x_train, y_train = CifarFeaturePersister.load('sobel_train', sample_size=train_size)
            x_test, y_test = CifarFeaturePersister.load('sobel_test', sample_size=test_size)
        elif feature_to_extract == "Canny":
            x_train, y_train = CifarFeaturePersister.load('canny_sigma_1.2_train', sample_size=train_size)
            x_test, y_test = CifarFeaturePersister.load('canny_sigma_1.2_test', sample_size=test_size)
        else:
            x_train, y_train = CifarDataset.take_sample_of_cifar_train(train_size)
            x_test, y_test = CifarDataset.take_sample_of_cifar_test(test_size)
        
        knn = KNearestNeighbor()

        knn.fit(x_train, y_train)
        metrics, confusion_matrix = knn.score(x_test, y_test, k=k)

        probs = knn.predict_prob(x_test, k=k)

        # Reset any existing plots and axis
        self.tl_figure.clear()
        self.tr_figure.clear()
        self.bl_figure.clear()

        # ROC Curve
        tl_axes = self.tl_figure.add_subplot()

        for i in range(10):
            tpr, fpr, thresholds = Metrics.roc_curve(y_test, probs[:, i], i)
            tl_axes.plot(fpr, tpr, label=CifarDatasetUtils.label_as_string(i))

        tl_axes.plot([0.0, 1.0], [0.0, 1.0], color='black', label='chance-line')


        tl_axes.set_title("ROC Curve for KNN")
        tl_axes.set_xlabel("False Positive Rate")
        tl_axes.set_ylabel("True Positive Rate")
        tl_axes.legend()

        # Precision Recall Bars
        precision_val = list(zip(*metrics['precision']))[1]
        recall_val = list(zip(*metrics['recall']))[1]
        tr_axes = self.tr_figure.add_subplot()
        tr_axes.bar(np.arange(len(precision_val)) - 0.2, precision_val, 0.40, label='Precision')
        tr_axes.bar(np.arange(len(recall_val)) + 0.2, recall_val, 0.40, label='Recall')

        tr_axes.set_title("Precision and Recall")
        classifications = CifarDatasetUtils.classifications()
        tr_axes.set_xticks(range(len(classifications)), classifications, size='small')
        tr_axes.set_xlabel("Classification")
        tr_axes.legend()

        self.metrics_header.setText("Model Metrics:")
        self.metrics_accuracy.setText("Accuracy: {} \n".format(metrics['accuracy']))

        precision = np.array(metrics['precision'])[:,1]
        recall = np.array(metrics['recall'])[:,1]
        f1 = np.array(metrics['f1'])[:,1]

        self.metrics_precision.setText("Precision: {} \nAverage: {:.2f}".format(metrics['precision'], np.mean(precision)))
        self.metrics_recall.setText("Recall: {} \nAverage: {:.2f}".format(metrics['recall'], np.mean(recall)))
        self.metrics_f1.setText("F1: {} \nAverage: {:.2f}".format(metrics['f1'], np.mean(f1)))

        bl_axes = self.bl_figure.add_subplot()
        Metrics.plot_confusion_matrix(confusion_matrix, bl_axes)
        bl_axes.set_title("Confusion Matrix")

        self.tl_canvas.draw()
        self.tr_canvas.draw()
        self.bl_canvas.draw()
