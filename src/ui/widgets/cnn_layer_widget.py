import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QSpinBox, QWidget, QComboBox)
from model.cnn.layer.layer import Layer
from model.cnn.layer.conv import Conv
from model.cnn.layer.max_pooling import MaxPooling
from model.cnn.layer.dense import Dense

from ui.util.ui_utils import UiUtils

class CnnLayerWidget(QWidget):
    def __init__(self, layer:Layer = None, parent=None):
        super().__init__(parent)

        # Setup layer parameter
        layout = QHBoxLayout()

        self.layer_type_combo_box = QComboBox()
        self.layer_type_combo_box.addItems([
            'Conv',
            'Max Pool',
            'Dense'
        ])

        self.layer_type_combo_box.currentTextChanged.connect(self._setup_layer_parameter_layout)

        self.parameters_layout = QHBoxLayout()

        layout.addWidget(self.layer_type_combo_box)
        layout.addLayout(self.parameters_layout)
        layout.setContentsMargins(0,0,0,0)
        layout.addStretch()
        self.setLayout(layout)

        self._setup_layer_parameter_layout()

        # If a CNN layer is provided reflect this in the UI
        if layer is not None:
            self._set_ui_from_layer(layer)

    def _setup_layer_parameter_layout(self):
        # Remove all existing layout
        UiUtils.remove_all_widgets_from_layout(self.parameters_layout)

        if self.layer_type_combo_box.currentText() == 'Conv':

            print("Setting UI for Conv")
            self.conv_num_filters = QSpinBox()
            self.conv_num_filters.setRange(1, 256)

            self.conv_activation = QComboBox()
            self.conv_activation.addItems([
                'relu',
                'sigmoid'
            ])

            self.conv_filter_size = QSpinBox()
            self.conv_filter_size.setRange(1, 16)

            self.parameters_layout.addWidget(QLabel("Filters:"))
            self.parameters_layout.addWidget(self.conv_num_filters)
            self.parameters_layout.addWidget(QLabel("Filter Size:"))
            self.parameters_layout.addWidget(self.conv_filter_size)
            self.parameters_layout.addWidget(QLabel("Activation:"))
            self.parameters_layout.addWidget(self.conv_activation)

        elif self.layer_type_combo_box.currentText() == 'Max Pool':

            print("Setting UI for Max Pool")
            self.max_pool_size = QSpinBox()
            self.max_pool_size.setRange(1, 16)

            self.parameters_layout.addWidget(QLabel("Pool Size:"))
            self.parameters_layout.addWidget(self.max_pool_size)
        elif self.layer_type_combo_box.currentText() == 'Dense':

            print("Setting UI for Dense")
            self.dense_num_units = QSpinBox()
            self.dense_num_units.setRange(1, 3072)

            self.dense_activation = QComboBox()
            self.dense_activation.addItems([
                'softmax',
                'relu',
                'sigmoid'
            ])

            self.parameters_layout.addWidget(QLabel("Output Units:"))
            self.parameters_layout.addWidget(self.dense_num_units)
            self.parameters_layout.addWidget(QLabel("Activation:"))
            self.parameters_layout.addWidget(self.dense_activation)
        else:
            raise Exception("Unsupported Layer type in CnnLayerWidget")

    def _set_ui_from_layer(self, layer:Layer):
        if isinstance(layer, Conv):
            self.layer_type_combo_box.setCurrentText('Conv')
            self.conv_num_filters.setValue(layer.num_filters)
            self.conv_activation.setCurrentText(layer.activation)
            self.conv_filter_size.setValue(layer.filter_shape[0])
        elif isinstance(layer, MaxPooling):
            self.layer_type_combo_box.setCurrentText('Max Pool')
            self.max_pool_size.setValue(layer.pooling_shape[0])
        elif isinstance(layer, Dense):
            self.layer_type_combo_box.setCurrentText('Dense')
            self.dense_num_units.setValue(layer.output_units)
            self.dense_activation.setCurrentText(layer.activation)
        else:
            raise Exception("Unsupported Layer type in CnnLayerWidget")

    def get_layer_from_ui(self):
        if self.layer_type_combo_box.currentText() == 'Conv':
            shape = self.conv_filter_size.value()
            return Conv(activation=self.conv_activation.currentText(), filters=self.conv_num_filters.value(), filter_shape=(shape,shape))
        elif self.layer_type_combo_box.currentText() == 'Max Pool':
            shape = self.max_pool_size.value()
            return MaxPooling(pooling_shape=(shape, shape))
        elif self.layer_type_combo_box.currentText() == 'Dense':
            return Dense(activation=self.dense_activation.currentText(), output_units=self.dense_num_units.value())
        else:
            raise Exception("Unsupported Layer type in CnnLayerWidget")