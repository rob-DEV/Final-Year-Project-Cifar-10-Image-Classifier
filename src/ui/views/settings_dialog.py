from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QCheckBox, QVBoxLayout

from ui.config.settings import Settings

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedWidth(300)
        self.setWindowTitle("Settings")

        self.close_btn = QDialogButtonBox(QDialogButtonBox.Ok)
        self.close_btn.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel("Settings")
        
        darkmode_state = Settings.get("SETTINGS", "dark_mode")
        
        self.darkmode_checkbox = QCheckBox("Darkmode")
        self.darkmode_checkbox.setChecked(bool(darkmode_state))
        self.darkmode_checkbox.toggled.connect(self.darkmode_checkbox_toggled)

        self.layout.addWidget(message)
        self.layout.addWidget(self.darkmode_checkbox)
        self.layout.addWidget(self.close_btn)
        self.setLayout(self.layout)

    def darkmode_checkbox_toggled(self):
        from ui.application import Application
        from ui.config.settings import Settings

        isChecked = self.darkmode_checkbox.isChecked()
        Settings.set("SETTINGS", "dark_mode", str(isChecked))
        Application.set_application_style()
