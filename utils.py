from PyQt5.QtWidgets import QFileDialog
from datetime import date
import pandas as pd

def select_file(obj):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "","Excel File (*.xlsx)", options=options)
    obj.lineEdit.setText(fileName)
    
def train(obj):
    dt_from = obj.train_dtfrom.text()
    dt_to = obj.train_dtto.text()
    process_data(obj, dt_from, dt_to)
    
def forecast(obj):
    dt_from = obj.test_dtfrom.text()
    dt_to = obj.test_dtto.text()
    process_data(obj, dt_from, dt_to)

def process_data(obj, dt_from, dt_to):
    filename = obj.lineEdit.text()

    from_split = dt_from.split("/")
    dt_from = date(int(from_split[0]), int(from_split[1]), int(from_split[2]))
    to_split = dt_to.split("/")
    dt_to = date(int(to_split[0]), int(to_split[1]), int(to_split[2]))
    
    df = pd.read_excel(filename, header=None)
    df["date"] = df.apply(lambda x: date(int(x[0]), int(x[1]), int(x[2])), axis=1)
    df = df.loc[(df["date"] >= dt_from) & ((df["date"] <= dt_to))]
    df = df.drop(columns=[0, 1, 2, 3, "date"])
    
    return df