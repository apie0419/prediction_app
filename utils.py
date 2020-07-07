import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from tensorflow.python.client import device_lib
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressBar, QDialog
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from datetime import date
from predictor import get_model

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def select_file(obj):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "","Excel File (*.xlsx)", options=options)
    obj.setText(fileName)
    
def train(obj):
    dt_from = obj.train_dtfrom.text()
    dt_to = obj.train_dtto.text()
    data, target = process_data(obj, dt_from, dt_to)

def forecast(obj):
    dt_from = obj.test_dtfrom.text()
    dt_to = obj.test_dtto.text()
    data, target = process_data(obj, dt_from, dt_to)
    model = get_model(obj.model_cbbox.currentText())
    predict, target, loss = model.test(data, target, obj.progress_layout)
    sc = MplCanvas(None, width=5, height=4, dpi=100)
    df = pd.DataFrame({
        "predict": predict.flatten(),
        "target": target.flatten()
    })

    df.plot(ax=sc.axes)
    toolbar = NavigationToolbar(sc, None)
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(toolbar)
    layout.addWidget(sc)
    obj.graph_widget.setLayout(layout)
    obj.rmse_loss_edit.setText(str(loss) + "%")

def process_data(obj, dt_from, dt_to):
    data_filepath = obj.data_file_edit.text()
    target_filepath = obj.target_file_edit.text()

    from_split = dt_from.split("/")
    dt_from = date(int(from_split[0]), int(from_split[1]), int(from_split[2]))
    to_split = dt_to.split("/")
    dt_to = date(int(to_split[0]), int(to_split[1]), int(to_split[2]))
    
    df = pd.read_excel(data_filepath, header=None)
    df["date"] = df.apply(lambda x: date(int(x[0]), int(x[1]), int(x[2])), axis=1)
    df = df.loc[(df["date"] >= dt_from) & ((df["date"] <= dt_to))]
    df = df.drop(columns=["date"])

    df2 = pd.read_excel(target_filepath, header=None)
    df2["date"] = df2.apply(lambda x: date(int(x[0]), int(x[1]), int(x[2])), axis=1)
    df2 = df2.loc[(df2["date"] >= dt_from) & ((df2["date"] <= dt_to))]
    df2 = df2.drop(columns=["date"])
    
    return df.values, df2.values

def select_gpu_device(obj):
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name[-1] for x in local_device_protos if x.device_type == 'GPU']
    if len(gpus) > 0:
        for idx, gpu in enumerate(gpus):
            obj.gpu_cbbox.setItemText(idx, str(gpu))
    else:
        msg = QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText("Cannot Detect GPU Devices")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
        obj.gpu_ckbox.setChecked(False)