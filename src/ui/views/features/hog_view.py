from dataset.cifar_dataset import CifarDataset
from feature_extraction.hog import HOG
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as GraphCanvas
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)


class HogView(QWidget):

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
        self.parameter_layout.addWidget(QLabel("HOG Parameters"))

        parameter_form_layout = QVBoxLayout()

        info_label = QLabel(
            "Selects a sample of CIFAR-10 images, extracts a HOG feature vector then builds an visual image of the generated vector.\n")
        info_label.setWordWrap(True)

        block_size_info = QLabel(
            "Block size: \nSets the width and height for each sub region of the image, a smaller block size means more blocks per image hence a longer feature extraction time.")
        block_size_info.setWordWrap(True)
        self.hog_block_size = QSpinBox()
        self.hog_block_size.setRange(0, 16)
        self.hog_block_size.setValue(8)

        run_button = QPushButton("Extract")
        run_button.clicked.connect(self.extract_feature)

        parameter_form_layout.addWidget(info_label)
        parameter_form_layout.addWidget(block_size_info)
        parameter_form_layout.addWidget(self.hog_block_size)
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
        hog = HOG(block_size=self.hog_block_size.value())
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
            hog_vector = hog.extract(cifar_images[i])
            hog_image = HOG.build_hog_image(hog_vector)

            axes = self.final_feature_figure.add_subplot(221 + i)
            axes.imshow(hog_image, cmap='gray')
            axes.set_xticks([])
            axes.set_yticks([])

        self.initial_images_canvas.draw()
        self.final_feature_canvas.draw()
