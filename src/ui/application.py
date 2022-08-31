import sys

from PyQt5.QtCore import QFile, QTextStream
from PyQt5.QtWidgets import QApplication

from ui.views.main_window import MainWindow


class Application:
    _app = QApplication(sys.argv)
    _main_window = MainWindow()

    @staticmethod
    def set_application_style() -> None:
        from ui.config.settings import Settings
        if Settings.get("SETTINGS", "dark_mode") == "True":
            # Open and apply dark mode styling
            file = QFile("src\\ui\\themes\\dark.qss")
            file.open(QFile.ReadOnly | QFile.Text)
            stream = QTextStream(file)
            Application._app.setStyleSheet(stream.readAll())
        else:
            # Standard light mode
            Application._app.setStyleSheet("")

    def run() -> None:
        # Center and show window
        Application._main_window.showMaximized()
        sys.exit(Application._app.exec_())
