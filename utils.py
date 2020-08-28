import pandas as pd
import numpy as np

from tensorflow.python.client import device_lib
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5 import QtWidgets
from datetime import date
from predictor import get_model
from PyQt5.QtCore import pyqtSignal, QThread

def select_file(obj):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "","Excel File (*.xlsx)", options=options)
    obj.setText(fileName)

class train(QThread):

    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    graph = pyqtSignal(object, float, bool)
    status = pyqtSignal(bool)

    def __init__(self, obj):
        super(train, self).__init__()
        self.obj = obj

    def run(self):
        self.status.emit(True)
        dt_from = self.obj.train_dtfrom.text()
        dt_to = self.obj.train_dtto.text()
        data, target = process_data(self.obj, dt_from, dt_to)
        
        if type(data) == type(None) or type(target) == type(None):
            self.error.emit("Please Select Input and Output Files")
            return

        if len(data) == 0:
            self.error.emit("Please Select Correct Date Range")
            return

        use_gpu = self.obj.gpu_ckbox.isChecked()
        device = None
        if use_gpu:
            device = self.obj.gpu_cbbox.currentText()

        mode = None
        if self.obj.hourahead_btn.isChecked():
            mode = 1
        elif self.obj.dayahead_btn.isChecked():
            mode = 2

        modelname = self.obj.model_cbbox.currentText()
        model = get_model(modelname)
        model.init_model(use_gpu, device, mode)
        

        if modelname == "TCN":
            
            for losses in model.train(data, target):
                
                loss = round(losses.mean(), 2)

                df = pd.DataFrame({
                    "loss": losses.flatten(),
                })
                p = round(losses.shape[0] / 100 * 100.)

                self.graph.emit(df, loss, True)
                self.progress.emit(p)

        else:
            loss = model.train(data, target, self.progress)

            self.graph.emit(None, loss, True)


class forecast(QThread):

    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    graph = pyqtSignal(object, float, bool)
    status = pyqtSignal(bool)

    def __init__(self, obj):
        super(forecast, self).__init__()
        self.obj = obj

    def run(self):
        self.status.emit(False)
        dt_from = self.obj.test_dtfrom.text()
        dt_to = self.obj.test_dtto.text()
        use_gpu = self.obj.gpu_ckbox.isChecked()
        device = None
        mode = None
        if use_gpu:
            device = self.obj.gpu_cbbox.currentText()
        if self.obj.hourahead_btn.isChecked():
            mode = 1
        elif self.obj.dayahead_btn.isChecked():
            mode = 2

        data, target = process_data(self.obj, dt_from, dt_to)

        if type(data) == type(None) or type(target) == type(None):
            self.error.emit("Please Select Input and Output Files")
            return

        if len(data) == 0:
            self.error.emit("Please Select Correct Date Range")
            return

        modelname = self.obj.model_cbbox.currentText()
        model = get_model(modelname)
        model.init_model(use_gpu, device, mode)
        predict, target, loss = model.test(data, target, self.progress)

        if type(predict) == type(None) or type(target) == type(None) or type(loss) == type(None):
            self.error.emit("Please Train Correspondind model")
            return
            
        df = pd.DataFrame({
            "predict": predict.flatten(),
            "target": target.flatten()
        })
        self.graph.emit(df, loss, False)

def process_data(obj, dt_from, dt_to):
    data_filepath = obj.data_file_edit.text()
    target_filepath = obj.target_file_edit.text()
    if data_filepath == "" or target_filepath == "":
        return None, None

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
            obj.gpu_cbbox.setEnabled(True)
    else:
        box = QMessageBox()
        box.setWindowTitle("Error")
        box.setText("Cannot Detect GPU Devices")
        box.setIcon(QMessageBox.Critical)
        box.exec_()
        obj.gpu_ckbox.setChecked(False)