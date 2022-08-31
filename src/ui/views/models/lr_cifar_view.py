import numpy as np
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from feature_extraction.grayscale import Grayscale
from model.knn.k_nearest_neighbour import KNearestNeighbor
from model.logistic_regression.one_vs_all import Cifar10OneVsAll
from model.scoring.metrics import Metrics
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QDoubleSpinBox,
                             QLabel, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)
from persist.cifar_feature_persister import CifarFeaturePersister
from persist.model_persister import ModelPersister
from ui.util.ui_utils import UiUtils


class LogisticRegressionCifarView(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        parameter_layout = self.setup_parameter_layout()
        evaluation_layout = self.setup_evaluation_layout()

        layout.addLayout(parameter_layout, stretch=0)
        layout.addLayout(evaluation_layout, stretch=4)

        
        # Run the model initially to populate graphs
        # self.evaluate_model()
        

    def setup_parameter_layout(self):
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addSpacing(10)
        self.parameter_layout.addWidget(
            QLabel('Logistic Regression Parameters'))

        parameter_form_layout = QFormLayout()

        self.lr_combo_box = QComboBox()
        self.lr_combo_box.addItems([
            'HOG',
            'Grayscale',
            'RGB',
            'Histogram',
            'Sobel',
            'Canny'
        ])

        self.lr_train_size_spinbox = QSpinBox()
        self.lr_train_size_spinbox.setRange(1, 50000)
        self.lr_train_size_spinbox.setValue(100)

        self.lr_test_size_spinbox = QSpinBox()
        self.lr_test_size_spinbox.setRange(1, 10000)
        self.lr_test_size_spinbox.setValue(100)

        self.learning_rate_double_spinbox = QDoubleSpinBox()
        self.learning_rate_double_spinbox.setRange(0.001, 1.0)
        self.learning_rate_double_spinbox.setDecimals(4)
        self.learning_rate_double_spinbox.setValue(0.01)
        self.learning_rate_double_spinbox.setSingleStep(0.001)

        self.iteration_spin_box = QSpinBox()
        self.iteration_spin_box.setRange(1, 20000)
        self.iteration_spin_box.setValue(10)

        self.threshold_double_spinbox = QDoubleSpinBox()
        self.threshold_double_spinbox.setRange(0, 1)
        self.threshold_double_spinbox.setDecimals(3)
        self.threshold_double_spinbox.setValue(0.5)
        self.threshold_double_spinbox.setSingleStep(0.001)

        parameter_form_layout.addRow(
            'Comparison Feature: ', self.lr_combo_box)
        parameter_form_layout.addRow(
            'Train Data Size: ', self.lr_train_size_spinbox)
        parameter_form_layout.addRow(
            'Test Data Size: ', self.lr_test_size_spinbox)
        parameter_form_layout.addRow(
            'Learning Rate: ', self.learning_rate_double_spinbox)
        parameter_form_layout.addRow(
            'Iteration Count: ', self.iteration_spin_box)
        parameter_form_layout.addRow(
            'Threshold: ', self.threshold_double_spinbox)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.evaluate_model)
        save_button = QPushButton("Save Model")
        save_button.clicked.connect(self.save_model)
        parameter_form_layout.addRow(run_button)
        parameter_form_layout.addRow(save_button)

        self.parameter_layout.addLayout(parameter_form_layout)
        self.parameter_layout.addStretch()

        self.parameter_layout.addLayout(parameter_form_layout)
        self.parameter_layout.addStretch()

        return self.parameter_layout

    
    def setup_evaluation_layout(self):
        self.evaluation_layout = QVBoxLayout()
        self.evaluation_layout.addSpacing(10)
        self.evaluation_layout.addWidget(QLabel('Model Evaluation'))

        self.evaluation_graph_layout = QGridLayout()
        self.evaluation_graph_layout.setRowStretch(0, 3)
        self.evaluation_graph_layout.setRowStretch(1, 2)

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
        self.evaluation_graph_layout.addWidget(self.tr_canvas, 0, 1)
        self.evaluation_graph_layout.addWidget(self.bl_canvas, 1, 0)
        self.evaluation_graph_layout.addLayout(self.metrics_layout, 1, 1)

        self.evaluation_results_layout = QHBoxLayout()

        self.evaluation_layout.addLayout(self.evaluation_graph_layout)
        self.evaluation_layout.addLayout(self.evaluation_results_layout)

        self.evaluation_layout.addStretch()

        return self.evaluation_layout

    def evaluate_model(self):
        lr_feature = self.lr_combo_box.currentText()
        print("Running LR Model on CIFAR using {}...".format(lr_feature))

        train_size = self.lr_train_size_spinbox.value()
        test_size = self.lr_test_size_spinbox.value()

        feature_to_extract = self.lr_combo_box.currentText()

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
        x_train = CifarDatasetUtils.normalize_data_min_max_scale(x_train)
        x_test = CifarDatasetUtils.normalize_data_min_max_scale(x_test)
        
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        
        self.one_vs_all = Cifar10OneVsAll(self.learning_rate_double_spinbox.value(), self.iteration_spin_box.value())
        self.model_name = "lr/one_vs_all_lr_{}_iter_{}_feature_{}".format(self.learning_rate_double_spinbox.value(), self.iteration_spin_box.value(), lr_feature)

        history = self.one_vs_all.fit(x_train, y_train)

        metrics, confusion_matrix = self.one_vs_all.score(x_test, y_test)

        # Reset any existing plots and axis
        self.tl_figure.clear()
        self.tr_figure.clear()
        self.bl_figure.clear()

        UiUtils.plot_history_metric_multi_classifier(history, 'train_loss', self.tl_figure, "Training Loss")
        UiUtils.plot_history_metric_multi_classifier(history, 'train_accuracy', self.bl_figure, "Training Accuracy")
        # UiUtils.plot_history_metric(history, 'validation_loss', self.bl_figure, "Validation Loss")
        # UiUtils.plot_history_metric(history, 'validation_accuracy', self.br_figure, "Validation Accuracy")


        self.metrics_header.setText("Model Metrics:")
        self.metrics_accuracy.setText("Accuracy: {} \n".format(metrics['accuracy']))

        precision = np.array(metrics['precision'])
        recall = np.array(metrics['recall'])
        f1 = np.array(metrics['f1'])

        self.metrics_precision.setText("Precision: {} \nAverage: {:.2f}".format(metrics['precision'], np.mean(precision)))
        self.metrics_recall.setText("Recall: {} \nAverage: {:.2f}".format(metrics['recall'], np.mean(recall)))
        self.metrics_f1.setText("F1: {} \nAverage: {:.2f}".format(metrics['f1'], np.mean(f1)))

        bl_axes = self.tr_figure.add_subplot()
        Metrics.plot_confusion_matrix(confusion_matrix, bl_axes)
        bl_axes.set_title("Confusion Matrix")

        self.tl_canvas.draw()
        self.tr_canvas.draw()
        self.bl_canvas.draw()

    def save_model(self):
        if self.one_vs_all is not None:
            ModelPersister.persist(self.one_vs_all,self.model_name)
