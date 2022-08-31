import numpy as np
from dataset.cifar_dataset import CifarDataset
from feature_extraction.histogram import ColorHistogram
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

class ColorHistogramView(QWidget):

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
        self.parameter_layout.addWidget(QLabel('Histogram Image Selection'))

        parameter_form_layout = QVBoxLayout()

        info_label = QLabel(
            "Selects a sample of CIFAR-10 images and extracts their colour histogram.\n")
        info_label.setWordWrap(True)

        info_label_histo = QLabel(
            "Each Cifar image has 3 color channels (RBG) per pixel, each channel has 256 different intensities this extraction technique takes each channel and measures the amount of pixels at each level.")
        info_label_histo.setWordWrap(True)

        run_button = QPushButton("Extract")
        run_button.clicked.connect(self.extract_feature)

        parameter_form_layout.addWidget(info_label)
        parameter_form_layout.addWidget(info_label_histo)
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

        channel_name = {
            0: 'Red',
            1: 'Green',
            2: 'Blue',
        }

        # Plot each images colour histogram
        for i in range(4):
            histo_extractor = ColorHistogram()
            histo = histo_extractor.extract(cifar_images[i])

            axes = self.final_feature_figure.add_subplot(221 + i)
            axes.plot(np.arange(256), histo[0],
                      color=channel_name.get(0).lower())
            axes.plot(np.arange(256), histo[1],
                      color=channel_name.get(1).lower())
            axes.plot(np.arange(256), histo[2],
                      color=channel_name.get(2).lower())
            axes.set_xlabel('Bins')
            axes.set_ylabel('Number of Pixels')

        self.initial_images_canvas.draw()
        self.final_feature_canvas.draw()
