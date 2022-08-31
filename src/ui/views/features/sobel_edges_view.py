from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from PyQt5.QtWidgets import (QHBoxLayout, QSpinBox,
                             QLabel, QPushButton, QVBoxLayout,
                             QWidget)
from dataset.cifar_dataset import CifarDataset
from feature_extraction.edge_extraction import SobelEdgeExtraction


class SobelEdgesView(QWidget):

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
        self.parameter_layout.addWidget(QLabel('Sobel Edge Parameters'))

        parameter_form_layout = QVBoxLayout()

        info_label = QLabel(
            "Selects a sample of CIFAR-10 images and extracts their edges with a given inclusion threshold.")
        info_label.setWordWrap(True)

        info_label_edge = QLabel(
            "A simple form of edge extraction whereby the image noise is first reduced, then a series of masks are convolved across the image to extract the gradients. These are then combined using trigonometry to form the edge data.")
        info_label_edge.setWordWrap(True)

        noise_reduction_info = QLabel(
            "Noise reduction blur:\nFactor of the initial image blur used to reduce image noise, if this is too high too much detail will be lost.")
        noise_reduction_info.setWordWrap(True)
        self.noise_reduction_factor = QSpinBox()
        self.noise_reduction_factor.setRange(0, 6)
        self.noise_reduction_factor.setValue(3)

        edge_threshold = QLabel(
            "Edge discount threshold:\nThreshold below which edge data will be discounted, aiming to eliminate weak edge data. This is a limited approach and NMS would be a better option. See Canny EE.")
        edge_threshold.setWordWrap(True)
        self.pixel_threshold_spinbox = QSpinBox()
        self.pixel_threshold_spinbox.setRange(0, 255)
        self.pixel_threshold_spinbox.setValue(50)

        run_button = QPushButton("Extract")
        run_button.clicked.connect(self.extract_feature)

        parameter_form_layout.addWidget(info_label)
        parameter_form_layout.addWidget(info_label_edge)
        parameter_form_layout.addSpacing(10)
        parameter_form_layout.addWidget(noise_reduction_info)
        parameter_form_layout.addWidget(self.noise_reduction_factor)
        parameter_form_layout.addWidget(edge_threshold)
        parameter_form_layout.addWidget(self.pixel_threshold_spinbox)
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

        self.initial_images_canvas = GraphCanvas(self.initial_images_figure)
        self.final_feature_canvas = GraphCanvas(self.final_feature_figure)

        intial_final_horizontal_layout.addWidget(self.initial_images_canvas)
        intial_final_horizontal_layout.addWidget(self.final_feature_canvas)

        self.evaluation_layout.addStretch()

        return self.evaluation_layout

    def extract_feature(self):
        cifar_images, _ = CifarDataset.take_sample_of_cifar_train(4)
        # Populate images to grid
        self.initial_images_figure.clear()
        self.final_feature_figure.clear()

        # 2x2 Grid of initial CIFAR-10 images
        for i in range(4):
            axes = self.initial_images_figure.add_subplot(221 + i)
            axes.imshow(cifar_images[i])
            axes.set_xticks([])
            axes.set_yticks([])

        # Extract edges and show
        for i in range(4):
            edge_extractor = SobelEdgeExtraction(noise_mask_size=self.noise_reduction_factor.value(), pixel_threshold=self.pixel_threshold_spinbox.value())
            edges = edge_extractor.extract(cifar_images[i])

            axes = self.final_feature_figure.add_subplot(221 + i)
            axes.imshow(edges, cmap='gray')
            axes.set_xticks([])
            axes.set_yticks([])

        self.initial_images_canvas.draw()
        self.final_feature_canvas.draw()
