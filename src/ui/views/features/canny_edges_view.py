from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from PyQt5.QtWidgets import (QHBoxLayout, QSpinBox, QDoubleSpinBox,
                             QLabel, QPushButton, QVBoxLayout,
                             QWidget)
import numpy as np
from dataset.cifar_dataset import CifarDataset
from feature_extraction.canny_edge_extraction import CannyEdgeExtraction


class CannyEdgesView(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        parameter_layout = self.setup_parameter_layout()
        evaluation_layout = self.setup_evaluation_layout()

        layout.addLayout(parameter_layout, stretch=0)
        layout.addLayout(evaluation_layout, stretch=5)

        self.extract_feature()

    def setup_parameter_layout(self):
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.addSpacing(10)
        self.parameter_layout.addWidget(QLabel('Canny Edge Parameters'))

        parameter_form_layout = QVBoxLayout()

        info_label = QLabel(
            "Selects a sample of CIFAR-10 images and extracts their edges.\n")
        info_label.setWordWrap(True)

        info_label_edge = QLabel(
            "A more effective edge extraction technique. It acheives this by using non maximum supression which eliminates pixels that do not define a \"strong\" edge, thereby reducing unwanted noise.")
        info_label_edge.setWordWrap(True)

        self.sigma_double_spinbox = QDoubleSpinBox()
        self.sigma_double_spinbox.setRange(0, 3.5)
        self.sigma_double_spinbox.setDecimals(3)
        self.sigma_double_spinbox.setValue(0.4)
        self.sigma_double_spinbox.setSingleStep(0.01)

        run_button = QPushButton("Extract")
        run_button.clicked.connect(self.extract_feature)

        parameter_form_layout.addWidget(info_label)
        parameter_form_layout.addWidget(info_label_edge)
        parameter_form_layout.addWidget(self.sigma_double_spinbox)
        parameter_form_layout.addWidget(run_button)

        self.parameter_layout.addLayout(parameter_form_layout)
        self.parameter_layout.addStretch()

        return self.parameter_layout

    def setup_evaluation_layout(self):
        self.evaluation_layout = QVBoxLayout()
        self.evaluation_layout.addSpacing(10)
        self.evaluation_layout.addWidget(QLabel('Feature Extraction'))

        intial_final_horizontal_layout = QHBoxLayout()
        self.evaluation_layout.addLayout(intial_final_horizontal_layout)

        self.initial_images_figure = plt.figure()
        self.final_feature_figure = plt.figure()
        self.gaussian_figure = plt.figure()

        self.initial_images_canvas = GraphCanvas(self.initial_images_figure)
        self.final_feature_canvas = GraphCanvas(self.final_feature_figure)
        self.gaussian_canvas = GraphCanvas(self.gaussian_figure)

        intial_final_horizontal_layout.addWidget(self.initial_images_canvas)
        intial_final_horizontal_layout.addWidget(self.final_feature_canvas)
        intial_final_horizontal_layout.addWidget(self.gaussian_canvas)

        self.evaluation_layout.addStretch()

        return self.evaluation_layout

    def extract_feature(self):
        cifar_images, _ = CifarDataset.take_sample_of_cifar_train(4)
        # Populate images to grid
        self.initial_images_figure.clear()
        self.gaussian_figure.clear()
        self.final_feature_figure.clear()

        # 2x2 Grid of initial CIFAR-10 images
        for i in range(4):
            axes = self.initial_images_figure.add_subplot(221 + i)
            axes.imshow(cifar_images[i])
            axes.set_xticks([])
            axes.set_yticks([])

        # Extract edges and show
        for i in range(4):
            canny_extractor = CannyEdgeExtraction(sigma=self.sigma_double_spinbox.value())
            edges, gaussian_kernel = canny_extractor.extract(
                cifar_images[i], return_gaussian=True)

            axes = self.final_feature_figure.add_subplot(221 + i)
            axes.imshow(edges, cmap='gray')
            axes.set_xticks([])
            axes.set_yticks([])

        # Show how the gaussian kernel changes with various sigma values
        x = np.arange(gaussian_kernel.shape[0])
        y = np.arange(gaussian_kernel.shape[1])

        X, Y = np.meshgrid(x, y)

        axes = self.gaussian_figure.add_subplot(projection='3d')
        axes.plot_surface(X, Y, gaussian_kernel, rstride=1,
                          cstride=1, cmap='winter', edgecolor='none')
        axes.set_title("Gaussian Kernel at sigma={}".format(
            self.sigma_double_spinbox.value()))

        self.initial_images_canvas.draw()
        self.gaussian_canvas.draw()
        self.final_feature_canvas.draw()
