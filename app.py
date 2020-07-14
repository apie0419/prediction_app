import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QProgressBar, QMessageBox
from utils import *

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Ui_MainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setEnabled(True)
        mainWindow.resize(871, 656)
        mainWindow.setMinimumSize(QtCore.QSize(871, 656))
        mainWindow.setMaximumSize(QtCore.QSize(871, 656))
        mainWindow.setSizeIncrement(QtCore.QSize(871, 656))
        mainWindow.setBaseSize(QtCore.QSize(871, 656))
        font = QtGui.QFont()
        font.setFamily("Arial")
        mainWindow.setFont(font)
        mainWindow.setAcceptDrops(False)
        mainWindow.setStatusTip("")
        mainWindow.setAutoFillBackground(False)
        mainWindow.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 271, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.data_file_btn = QtWidgets.QPushButton(self.centralwidget)
        self.data_file_btn.setGeometry(QtCore.QRect(230, 50, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setItalic(True)
        self.data_file_btn.setFont(font)
        self.data_file_btn.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: black;\n"
"border-radius: 5px;\n"
"border-width: 1px;\n"
"border-style: outset;\n"
"padding: 4px;")
        self.data_file_btn.setObjectName("data_file_btn")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 50, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.data_file_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.data_file_edit.setEnabled(True)
        self.data_file_edit.setGeometry(QtCore.QRect(80, 50, 141, 21))
        self.data_file_edit.setReadOnly(True)
        self.data_file_edit.setObjectName("data_file_edit")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 271, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gpu_ckbox = QtWidgets.QCheckBox(self.centralwidget)
        self.gpu_ckbox.setEnabled(True)
        self.gpu_ckbox.setGeometry(QtCore.QRect(20, 150, 121, 21))
        self.gpu_ckbox.setCheckable(True)
        self.gpu_ckbox.setChecked(False)
        self.gpu_ckbox.setObjectName("gpu_ckbox")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 180, 271, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 200, 271, 111))
        self.tabWidget.setObjectName("tabWidget")
        self.train_tab = QtWidgets.QWidget()
        self.train_tab.setObjectName("train_tab")
        self.label_5 = QtWidgets.QLabel(self.train_tab)
        self.label_5.setGeometry(QtCore.QRect(10, 20, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.train_tab)
        self.label_6.setGeometry(QtCore.QRect(10, 50, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.train_dtfrom = QtWidgets.QDateEdit(self.train_tab)
        self.train_dtfrom.setGeometry(QtCore.QRect(90, 20, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.train_dtfrom.setFont(font)
        self.train_dtfrom.setAlignment(QtCore.Qt.AlignCenter)
        self.train_dtfrom.setProperty("showGroupSeparator", False)
        self.train_dtfrom.setCalendarPopup(True)
        self.train_dtfrom.setObjectName("train_dtfrom")
        self.train_dtto = QtWidgets.QDateEdit(self.train_tab)
        self.train_dtto.setGeometry(QtCore.QRect(90, 50, 151, 21))
        self.train_dtto.setAlignment(QtCore.Qt.AlignCenter)
        self.train_dtto.setCalendarPopup(True)
        self.train_dtto.setTimeSpec(QtCore.Qt.LocalTime)
        self.train_dtto.setObjectName("train_dtto")
        self.tabWidget.addTab(self.train_tab, "")
        self.test_tab = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("AR JULIAN")
        self.test_tab.setFont(font)
        self.test_tab.setObjectName("test_tab")
        self.label_7 = QtWidgets.QLabel(self.test_tab)
        self.label_7.setGeometry(QtCore.QRect(10, 20, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.test_tab)
        self.label_8.setGeometry(QtCore.QRect(10, 50, 61, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.test_dtfrom = QtWidgets.QDateEdit(self.test_tab)
        self.test_dtfrom.setGeometry(QtCore.QRect(90, 20, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.test_dtfrom.setFont(font)
        self.test_dtfrom.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.test_dtfrom.setAlignment(QtCore.Qt.AlignCenter)
        self.test_dtfrom.setCalendarPopup(True)
        self.test_dtfrom.setObjectName("test_dtfrom")
        self.test_dtto = QtWidgets.QDateEdit(self.test_tab)
        self.test_dtto.setGeometry(QtCore.QRect(90, 50, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.test_dtto.setFont(font)
        self.test_dtto.setAlignment(QtCore.Qt.AlignCenter)
        self.test_dtto.setCalendarPopup(True)
        self.test_dtto.setObjectName("test_dtto")
        self.tabWidget.addTab(self.test_tab, "")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(10, 320, 271, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.model_cbbox = QtWidgets.QComboBox(self.centralwidget)
        self.model_cbbox.setGeometry(QtCore.QRect(30, 360, 231, 22))
        self.model_cbbox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.model_cbbox.setObjectName("model_cbbox")
        self.model_cbbox.addItem("")
        self.model_cbbox.addItem("")
        self.method_cbbox = QtWidgets.QComboBox(self.centralwidget)
        self.method_cbbox.setGeometry(QtCore.QRect(30, 390, 231, 22))
        self.method_cbbox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.method_cbbox.setObjectName("method_cbbox")
        self.method_cbbox.addItem("")
        self.method_cbbox.addItem("")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 430, 271, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.train_btn = QtWidgets.QPushButton(self.centralwidget)
        self.train_btn.setGeometry(QtCore.QRect(30, 520, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.train_btn.setFont(font)
        self.train_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.train_btn.setStyleSheet("QPushButton {\n"
"    border-color:black;\n"
"    border-radius: 5px;\n"
"    border-width: 1px;\n"
"    border-style: outset;\n"
"    padding: 4px;\n"
"    background-color: rgb(116, 142, 230);\n"
"    color: white;\n"
"}\n"
"QPushButton::hover\n"
"{\n"
"     background-color : rgb(129, 159, 255);\n"
"    background-color : rgb(17, 247, 0);\n"
"}\n"
"QPushButton::pressed\n"
"{\n"
"     background-color : rgb(89, 110, 177);\n"
"    background-color : rgb(15, 190, 0);\n"
"}")
        self.train_btn.setObjectName("train_btn")
        self.forecast_btn = QtWidgets.QPushButton(self.centralwidget)
        self.forecast_btn.setGeometry(QtCore.QRect(160, 520, 91, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.forecast_btn.setFont(font)
        self.forecast_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.forecast_btn.setStyleSheet("QPushButton {\n"
"    border-color:black;\n"
"    border-radius: 5px;\n"
"    border-width: 1px;\n"
"    border-style: outset;\n"
"    padding: 4px;\n"
"    background-color: rgb(116, 142, 230);\n"
"    color: white;\n"
"}\n"
"QPushButton::hover\n"
"{\n"
"     background-color : rgb(129, 159, 255);\n"
"    background-color : rgb(17, 247, 0);\n"
"}\n"
"QPushButton::pressed\n"
"{\n"
"     background-color : rgb(89, 110, 177);\n"
"    background-color : rgb(15, 190, 0);\n"
"}")
        self.forecast_btn.setObjectName("forecast_btn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(280, 10, 20, 621))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.target_file_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.target_file_edit.setEnabled(True)
        self.target_file_edit.setGeometry(QtCore.QRect(80, 80, 141, 21))
        self.target_file_edit.setReadOnly(True)
        self.target_file_edit.setObjectName("target_file_edit")
        self.target_file_btn = QtWidgets.QPushButton(self.centralwidget)
        self.target_file_btn.setGeometry(QtCore.QRect(230, 80, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setItalic(True)
        self.target_file_btn.setFont(font)
        self.target_file_btn.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: black;\n"
"border-radius: 5px;\n"
"border-width: 1px;\n"
"border-style: outset;\n"
"padding: 4px;")
        self.target_file_btn.setObjectName("target_file_btn")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(0, 80, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.graph_widget = QtWidgets.QWidget(self.centralwidget)
        self.graph_widget.setGeometry(QtCore.QRect(300, 10, 561, 581))
        self.graph_widget.setObjectName("graph_widget")
        self.gpu_cbbox = QtWidgets.QComboBox(self.centralwidget)
        self.gpu_cbbox.setEnabled(False)
        self.gpu_cbbox.setGeometry(QtCore.QRect(160, 150, 69, 22))
        self.gpu_cbbox.setObjectName("gpu_cbbox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 580, 271, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.progress_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_layout.setObjectName("progress_layout")
        self.hourahead_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.hourahead_btn.setGeometry(QtCore.QRect(30, 470, 111, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.hourahead_btn.setFont(font)
        self.hourahead_btn.setChecked(True)
        self.hourahead_btn.setObjectName("hourahead_btn")
        self.dayahead_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.dayahead_btn.setGeometry(QtCore.QRect(160, 470, 111, 22))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.dayahead_btn.setFont(font)
        self.dayahead_btn.setObjectName("dayahead_btn")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(300, 590, 561, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.result_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.setObjectName("result_layout")
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)
        self.actionWind_Power = QtWidgets.QAction(mainWindow)
        self.actionWind_Power.setVisible(True)
        self.actionWind_Power.setObjectName("actionWind_Power")
        self.actionSolar_Power = QtWidgets.QAction(mainWindow)
        self.actionSolar_Power.setEnabled(False)
        self.actionSolar_Power.setObjectName("actionSolar_Power")
        self.actionExit = QtWidgets.QAction(mainWindow)
        self.actionExit.setObjectName("actionExit")

        self.retranslateUi(mainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "預測系統"))
        self.label.setText(_translate("mainWindow", "Import Data"))
        self.data_file_btn.setText(_translate("mainWindow", "Browse"))
        self.label_2.setText(_translate("mainWindow", "Data File"))
        self.label_3.setText(_translate("mainWindow", "Device Configuration"))
        self.gpu_ckbox.setText(_translate("mainWindow", "Use GPU"))
        self.label_4.setText(_translate("mainWindow", "Select Data Range"))
        self.label_5.setText(_translate("mainWindow", "From"))
        self.label_6.setText(_translate("mainWindow", "    To"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.train_tab), _translate("mainWindow", "Training"))
        self.label_7.setText(_translate("mainWindow", "From"))
        self.label_8.setText(_translate("mainWindow", "    To"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.test_tab), _translate("mainWindow", "Testing"))
        self.label_9.setText(_translate("mainWindow", "Choose Methods"))
        self.model_cbbox.setItemText(0, _translate("mainWindow", "TCN"))
        self.model_cbbox.setItemText(1, _translate("mainWindow", "XGBOOST"))
        self.method_cbbox.setItemText(0, _translate("mainWindow", "Deterministic Forecasting"))
        self.method_cbbox.setItemText(1, _translate("mainWindow", "Probabilistic Forecasting"))
        self.label_10.setText(_translate("mainWindow", "Lead Time"))
        self.train_btn.setText(_translate("mainWindow", "Train"))
        self.forecast_btn.setText(_translate("mainWindow", "Forecast"))
        self.target_file_btn.setText(_translate("mainWindow", "Browse"))
        self.label_13.setText(_translate("mainWindow", "Target File"))
        self.hourahead_btn.setText(_translate("mainWindow", "Hour-ahead"))
        self.dayahead_btn.setText(_translate("mainWindow", "Day-ahead"))
        self.actionWind_Power.setText(_translate("mainWindow", "Wind Power"))
        self.actionSolar_Power.setText(_translate("mainWindow", "Solar Power"))
        self.actionExit.setText(_translate("mainWindow", "Exit"))
        self.init()
        self.bind_func()

    def init(self):
        pbar_style = """
        QProgressBar{
            margin-top: 2px;
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
            font-size: 17px;
        }
        """
        self.pbar = QProgressBar()
        self.pbar.setStyleSheet(pbar_style)
        layout = self.progress_layout
        layout.addWidget(self.pbar)
        self.pbar.setValue(0)
        self.train_func = train(self)
        self.forecast_func = forecast(self)
        self.train_func.progress.connect(self.update_pbar)
        self.train_func.graph.connect(self.update_graph)
        self.train_func.error.connect(self.showerror)
        self.train_func.status.connect(self.showstatus)
        self.forecast_func.progress.connect(self.update_pbar)
        self.forecast_func.graph.connect(self.update_graph)
        self.forecast_func.error.connect(self.showerror)
        self.forecast_func.status.connect(self.showstatus)

    def bind_func(self):
        self.data_file_btn.clicked.connect(lambda: select_file(self.data_file_edit))
        self.target_file_btn.clicked.connect(lambda: select_file(self.target_file_edit))
        self.train_btn.clicked.connect(self.train_func.start)
        self.forecast_btn.clicked.connect(self.forecast_func.start)
        self.gpu_ckbox.clicked.connect(lambda: select_gpu_device(self))

    def update_pbar(self, value):
        self.pbar.setValue(value)

    def update_graph(self, df, loss, training=True):
        for i in reversed(range(self.result_layout.count())): 
            self.result_layout.itemAt(i).widget().setParent(None)

        label = QtWidgets.QLabel(self.centralwidget)
        label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        label.setFont(font)
        if training:
            label.setText(f"Training RMSE Loss: {loss}%")
        else:
            label.setText(f"Testing RMSE Loss: {loss}%")

        self.result_layout.addWidget(label)

        layout = self.graph_widget.layout()
        if layout == None:
            layout = QtWidgets.QVBoxLayout()
        else:
            for i in reversed(range(layout.count())): 
                layout.itemAt(i).widget().setParent(None)
        if type(df) != type(None):
            sc = MplCanvas(None, width=5, height=4, dpi=100)
            ax = df.plot(ax=sc.axes)
            if training:
                ax.set_xlabel("Epoch")
                ax.set_ylabel("RMSE Loss")
            toolbar = NavigationToolbar(sc, None)
            layout.addWidget(toolbar)
            layout.addWidget(sc)
        self.graph_widget.setLayout(layout)

    def showerror(self, msg):

        box = QMessageBox()
        box.setWindowTitle("Error")
        box.setText(msg)
        box.setIcon(QMessageBox.Critical)
        box.exec_()

    def showstatus(self, training=True):
        for i in reversed(range(self.result_layout.count())): 
            self.result_layout.itemAt(i).widget().setParent(None)

        label = QtWidgets.QLabel(self.centralwidget)
        label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        label.setFont(font)
        if training:
            label.setText("Training")
        else:
            label.setText("Testing")

        self.result_layout.addWidget(label)
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
