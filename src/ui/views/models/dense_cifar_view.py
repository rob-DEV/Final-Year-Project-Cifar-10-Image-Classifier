from random import Random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from PyQt5.QtWidgets import (QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QComboBox)
import numpy as np
from dataset.cifar_dataset import CifarDataset
from dataset.cifar_dataset_utils import CifarDatasetUtils
from model.cnn.layer.conv import Conv
from model.cnn.layer.dense import Dense
from model.cnn.sequential_model import SequentialModel
from model.scoring.metrics import Metrics
from persist.cifar_feature_persister import CifarFeaturePersister
from ui.util.ui_utils import UiUtils
from ui.widgets.cnn_layer_widget import CnnLayerWidget


class DenseCifarView(QWidget):

    def __init__(self, model: SequentialModel = None, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        parameter_layout = self.setup_parameter_layout()
        evaluation_layout = self.setup_evaluation_layout()

        layout.addLayout(parameter_layout, stretch=2)
        layout.addLayout(evaluation_layout, stretch=4)

        if model is not None:
            for layer in model.layers:
                self.add_nn_layer_widget(layer)


    def setup_parameter_layout(self):
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addSpacing(10)
        self.parameter_layout.addWidget(QLabel('CIFAR-10 Dense Parameters'))

        # Used to hold the network layers
        self.nn_layer_layout = QVBoxLayout()

        add_layer_button = QPushButton("Add Layer")
        remove_layer_button = QPushButton("Remove Layer")
        remove_all_button = QPushButton("Remove All")
        run_model_button = QPushButton("Run")

        add_layer_button.clicked.connect(self.add_nn_layer_widget)
        remove_layer_button.clicked.connect(self.remove_last_nn_widget)
        remove_all_button.clicked.connect(self.remove_all_nn_widgets)
        run_model_button.clicked.connect(self.run_model)

        self.parameter_layout.addLayout(self.nn_layer_layout)
        self.parameter_layout.addWidget(add_layer_button)
        self.parameter_layout.addWidget(remove_layer_button)
        self.parameter_layout.addWidget(remove_all_button)

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
        self.lr_train_size_spinbox.setRange(1, 10000)
        self.lr_train_size_spinbox.setValue(100)

        self.lr_test_size_spinbox = QSpinBox()
        self.lr_test_size_spinbox.setRange(1, 1000)
        self.lr_test_size_spinbox.setValue(100)

        self.learning_rate_double_spinbox = QDoubleSpinBox()
        self.learning_rate_double_spinbox.setRange(0.001, 1.0)
        self.learning_rate_double_spinbox.setDecimals(4)
        self.learning_rate_double_spinbox.setValue(0.01)
        self.learning_rate_double_spinbox.setSingleStep(0.001)

        self.epoch_spin_box = QSpinBox()
        self.epoch_spin_box.setRange(1, 100)
        self.epoch_spin_box.setValue(10)


        parameter_form_layout.addRow(
            'Comparison Feature: ', self.lr_combo_box)
        parameter_form_layout.addRow(
            'Train Data Size: ', self.lr_train_size_spinbox)
        parameter_form_layout.addRow(
            'Test Data Size: ', self.lr_test_size_spinbox)
        parameter_form_layout.addRow(
            'Learning Rate: ', self.learning_rate_double_spinbox)
        parameter_form_layout.addRow(
            'Epochs: ', self.epoch_spin_box)

        self.parameter_layout.addLayout(parameter_form_layout)
        self.parameter_layout.addWidget(run_model_button)
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

    def add_nn_layer_widget(self, layer):
        if layer is False:
            layer = Dense()

        self.nn_layer_layout.addWidget(CnnLayerWidget(layer))

    def remove_last_nn_widget(self):
        last_layer_widget_index = self.nn_layer_layout.count() - 1
        widget = self.nn_layer_layout.takeAt(last_layer_widget_index).widget()
        self.nn_layer_layout.removeWidget(widget)
        widget.setParent(None)

    def remove_all_nn_widgets(self):
        UiUtils.remove_all_widgets_from_layout(self.nn_layer_layout)

    def run_model(self):
        # Build a model from the layers and their parameters
        model = SequentialModel()

        nn_layer_widget_count = self.nn_layer_layout.count()
        for i in range(nn_layer_widget_count):
            nn_widget = self.nn_layer_layout.itemAt(i).widget() # type: CnnLayerWidget
            model.add(nn_widget.get_layer_from_ui())

        model.summarize()


        nn_feature = self.lr_combo_box.currentText()
        print("Building Neural Network using {} vectors...".format(nn_feature))

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

        # model fitting
        # Ensure input shape is specified to the first layer
        model.layers[0].input_shape = x_train.shape[1]
        model.compile()

        history = model.fit(x_train, y_train, epochs=self.epoch_spin_box.value(), learning_rate=self.learning_rate_double_spinbox.value())

        metrics, confusion_matrix = model.score(x_test, y_test)

        self.tl_figure.clear()
        self.tr_figure.clear()
        self.bl_figure.clear()

        UiUtils.plot_history_metric(history, 'train_loss', self.tl_figure, "Training Loss")
        UiUtils.plot_history_metric(history, 'train_accuracy', self.bl_figure, "Training Accuracy")

        bl_axes = self.tr_figure.add_subplot()
        Metrics.plot_confusion_matrix(confusion_matrix, bl_axes, dp=1)
        bl_axes.set_title("Confusion Matrix (x_test)")

        self.tl_canvas.draw()
        self.tr_canvas.draw()
        self.bl_canvas.draw()