from PyQt5.QtWidgets import QApplication
import sys

from gui import MainWindow



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow.MainWindow()
    sys.exit(app.exec_())