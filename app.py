from PyQt5 import QtWidgets, uic
import sys, os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
uifile_path = os.path.join(BASE_PATH, "assets/main.ui")

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(uifile_path, self)
        self.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()