from PyQt5.QtWidgets import (QApplication, QMainWindow, QButtonGroup, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QStackedLayout,
                             QVBoxLayout, QWidget)  
from PyQt5.QtCore import Qt
from ui.views.features.hog_view import HogView
from ui.views.features.sobel_edges_view import SobelEdgesView
from ui.views.features.canny_edges_view import CannyEdgesView
from ui.views.features.color_histogram_view import ColorHistogramView
from ui.views.models.dense_cifar_view import DenseCifarView
from ui.views.models.knn_view import KNearestNeighbourView
from ui.views.models.lr_cifar_view import LogisticRegressionCifarView
from ui.views.models.lr_simple_view import LogisticRegressionSimpleView
from ui.views.settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FYP: Cifar Image Classifier")

        # Create top menu bar
        self.setup_menu_bar_and_events()

        # Create the main application ui
        main_layout = self.setup_main_ui()

        # Set layout to the main widget of the window
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        print("Initialized Main Application Window")

    def setup_menu_bar_and_events(self):
        menu_bar = self.menuBar()
        menu = menu_bar.addMenu("Menu")
        settings = menu.addAction("Settings")
        about = menu.addAction("About")
        exit = menu.addAction("Exit")

        # Setup menu click events
        settings.triggered.connect(self.on_menu_settings_click)
        about.triggered.connect(self.on_menu_about_click)
        exit.triggered.connect(lambda: QApplication.exit())

    def setup_main_ui(self):
        # Create feature buttons
        self.feature_btn = [
            QPushButton('CIFAR-10 KNN'),
            QPushButton('Simple LR'),
            QPushButton('CIFAR-10 LR One Vs All'),
            QPushButton('CIFAR-10 Dense / CNN'),
            QPushButton('Colour Histogram'),
            QPushButton('Sobel Edges'),
            QPushButton('Canny Edges'),
            QPushButton('HOG')
        ]

        # Create a groups set of button for toggle vs parts of the UI are visible based on selected feature
        self.feature_btn_group = QButtonGroup()

        # Add buttons to button group with index
        for i in range(len(self.feature_btn)):
            self.feature_btn_group.addButton(self.feature_btn[i])

        self.feature_btn_group.buttonClicked.connect(
            self.on_button_click)

        main_layout = QHBoxLayout()

        feature_button_layout = QVBoxLayout()
        feature_button_layout.addSpacing(10)
        feature_button_layout.addWidget(QLabel('Models and Features'))

        for button in self.feature_btn:
            feature_button_layout.addWidget(button)

        feature_button_layout.addStretch()

        # Create the stack layout toggled by each button
        self.stacked_layout = QStackedLayout()

        # Load and store all feature view widgets
        self.knn_view = KNearestNeighbourView()
        self.lr_simple_test_view = LogisticRegressionSimpleView()
        self.lr_cifar_view = LogisticRegressionCifarView()
        self.cifar_dense_view = DenseCifarView()
        self.color_histogram_view = ColorHistogramView()
        self.sobel_edges_view = SobelEdgesView()
        self.canny_edges_view = CannyEdgesView()
        self.hog_view = HogView()

        self.stacked_layout.addWidget(self.knn_view)
        self.stacked_layout.addWidget(self.lr_simple_test_view)
        self.stacked_layout.addWidget(self.lr_cifar_view)
        self.stacked_layout.addWidget(self.cifar_dense_view)
        self.stacked_layout.addWidget(self.color_histogram_view)
        self.stacked_layout.addWidget(self.sobel_edges_view)
        self.stacked_layout.addWidget(self.canny_edges_view)
        self.stacked_layout.addWidget(self.hog_view)

        # Set to the first view (KNN)
        self.stacked_layout.setCurrentIndex(0)

        # Nest layouts in main layout
        main_layout.addLayout(feature_button_layout)
        main_layout.addLayout(self.stacked_layout)

        return main_layout

    def on_menu_settings_click(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
            )
        settings_dialog.exec()

    def on_menu_about_click(self):
        QMessageBox.about(
            self,
            "Final Year Project",
            "<p>Cifar Image Classifier:</p>"
            "<p>- HOG Implementation/Visualisation </p>"
            "<p>- SIFT Implementation/Visualisation </p>"
            "<p>- K Nearest Neighbor</p>"
            "<p>- One vs All Logisitic Regression</p>"
            "<p>- Convolutional Neural Network</p>",
        )

    def on_button_click(self, button):
        index = self.feature_btn.index(button)
        print("Switching to layout at index {}".format(index))
        self.stacked_layout.setCurrentIndex(index)
