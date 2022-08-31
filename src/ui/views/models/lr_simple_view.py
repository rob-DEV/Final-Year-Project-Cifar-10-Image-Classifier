import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from model.logistic_regression.lr import LogisticRegression
from PyQt5.QtWidgets import (QDoubleSpinBox, QFormLayout, QGridLayout,
                             QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

from model.scoring.metrics import Metrics


class LogisticRegressionSimpleView(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        parameter_layout = self.setup_parameter_layout()
        evaluation_layout = self.setup_evaluation_layout()

        layout.addLayout(parameter_layout, stretch=0)
        layout.addLayout(evaluation_layout, stretch=3)

        # Run the model initially to populate graphs
        self.evaluate_model()
        

    def setup_parameter_layout(self):
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addSpacing(10)
        self.parameter_layout.addWidget(QLabel('Logistic Regression Parameters'))

        parameter_form_layout = QFormLayout()

        self.lr_train_size_spinbox = QSpinBox()
        self.lr_train_size_spinbox.setRange(1, 10000)
        self.lr_train_size_spinbox.setValue(100)

        self.lr_test_size_spinbox = QSpinBox()
        self.lr_test_size_spinbox.setRange(1, 10000)
        self.lr_test_size_spinbox.setValue(100)

        self.learning_rate_double_spinbox = QDoubleSpinBox()
        self.learning_rate_double_spinbox.setRange(0.001, 5.0)
        self.learning_rate_double_spinbox.setDecimals(3)
        self.learning_rate_double_spinbox.setValue(0.001)
        self.learning_rate_double_spinbox.setSingleStep(0.001)

        self.iteration_spin_box = QSpinBox()
        self.iteration_spin_box.setRange(1,10000)
        self.iteration_spin_box.setValue(10)

        self.weight_spinbox = QDoubleSpinBox()
        self.weight_spinbox.setRange(0, 1)
        self.weight_spinbox.setDecimals(5)
        self.weight_spinbox.setSingleStep(0.001)
        self.weight_spinbox.setReadOnly(True)

        self.threshold_double_spinbox = QDoubleSpinBox()
        self.threshold_double_spinbox.setRange(0, 1)
        self.threshold_double_spinbox.setDecimals(3)
        self.threshold_double_spinbox.setValue(0.5)
        self.threshold_double_spinbox.setSingleStep(0.001)

        parameter_form_layout.addRow('Train Data Size: ', self.lr_train_size_spinbox)
        parameter_form_layout.addRow('Test Data Size: ', self.lr_test_size_spinbox)
        parameter_form_layout.addRow('Learning Rate: ', self.learning_rate_double_spinbox)
        parameter_form_layout.addRow('Iteration Count: ', self.iteration_spin_box)
        parameter_form_layout.addRow('Threshold: ', self.threshold_double_spinbox)
        parameter_form_layout.addRow('Final Weight Value: ', self.weight_spinbox)

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

        self.tl_figure = plt.figure()
        self.tr_figure = plt.figure()

        self.tl_canvas = GraphCanvas(self.tl_figure)
        self.tr_canvas = GraphCanvas(self.tr_figure)
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
        self.evaluation_graph_layout.addLayout(self.metrics_layout, 1, 0)

        self.evaluation_results_layout = QHBoxLayout()

        self.evaluation_layout.addLayout(self.evaluation_graph_layout)
        self.evaluation_layout.addLayout(self.evaluation_results_layout)

        self.evaluation_layout.addStretch()

        return self.evaluation_layout

    def evaluate_model(self):
        # Creating a simple dataset of positive and negative integers
        x_train = np.random.randint(-100000, 100000, self.lr_train_size_spinbox.value())
        y_train = np.array(x_train)
        y_train[y_train <= 0] = 0 
        y_train[y_train > 0] = 1

        x_test = np.random.randint(-100000, 100000, self.lr_test_size_spinbox.value())
        y_test = np.array(x_test)
        y_test[y_test <= 0] = 0 
        y_test[y_test > 0] = 1

        x_train_normalized = x_train.astype(np.float32) / 100000
        x_test_normalized = x_test.astype(np.float32) / 100000

        model = LogisticRegression(learning_rate=self.learning_rate_double_spinbox.value(), num_iterations=self.iteration_spin_box.value(), threshold=self.threshold_double_spinbox.value(), weight_init_type='random')
        model.fit(x_train_normalized, y_train)

        # Show the model's single weight
        self.weight_spinbox.setValue(np.max(model.w))


        probs = model.predict_prob(x_test_normalized)
        y_pred = model.predict(x_test_normalized)
        metrics, confusion_matrix = model.score(x_test_normalized, y_test)

        self.tl_figure.clear()
        self.tr_figure.clear()

        # Sigmoid Distribution
        tl_axes = self.tl_figure.add_subplot()
        tl_axes.scatter(x_test, probs, marker=',', s=.7)

        # Also plot the distribution of x_test
        # Postive class
        positive_indices = np.where(y_pred == 1)
        negative_indices = np.where(y_pred == 0)

        tl_axes.scatter(x_test[positive_indices], y_test[positive_indices], label='Postive Number')
        tl_axes.scatter(x_test[negative_indices], y_test[negative_indices], label='Negative Number')

        tl_axes.set_title("Logistic Regression")
        tl_axes.set_xlabel("Number Line (x)")
        tl_axes.set_ylabel("Sigmoid(x)")
        tl_axes.legend()

        # Confusion Matrix
        tr_axes = self.tr_figure.add_subplot()
        Metrics.plot_confusion_matrix(confusion_matrix, tr_axes)

        self.metrics_header.setText("Model Metrics:")
        self.metrics_accuracy.setText("Accuracy: {}".format(metrics['accuracy']))
        self.metrics_precision.setText("Precision: {}".format(metrics['precision']))
        self.metrics_recall.setText("Recall: {}".format(metrics['recall']))
        self.metrics_f1.setText("F1: {}".format(metrics['f1']))

        self.tl_canvas.draw()
        self.tr_canvas.draw()
