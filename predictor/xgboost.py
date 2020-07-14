import pandas as pd
import xgboost as xgb
import numpy as np
import os, math, pickle

from .utils import process_data
from sklearn import metrics

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(BASE_PATH, "weights")

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

class xgboost():
    def __init__(self):
        self.gamma            = 0
        self.lr               = 0.005
        self.max_depth        = 10
        self.min_child_weight = 10
        self.n_estimators     = 1000
        self.reg_alpha        = 0.5
        self.reg_lambda       = 0.5
        self.eval_metric      = 'rmse'
        self.objective        = 'reg:logistic'
        self.subsample        = 0.7

    def init_model(self, use_gpu=False, device=0, mode=1):
        self.mode = mode
        
        if self.mode == 1:
            self.timesteps = 24
        elif self.mode == 2:
            self.timesteps = 8
        if use_gpu:
            self.model = xgb.XGBRegressor(
                gamma=self.gamma,
                learning_rate=self.lr,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                n_estimators=self.n_estimators,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                eval_metric=self.eval_metric,
                objective=self.objective,
                subsample=self.subsample,
                gpu_id=device,
                tree_method='gpu_hist',
                n_jobs=-1,
                seed=50
            )
        else:
            self.model = xgb.XGBRegressor(
                gamma=self.gamma,
                learning_rate=self.lr,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                n_estimators=self.n_estimators,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                eval_metric=self.eval_metric,
                objective=self.objective,
                subsample=self.subsample,
                n_jobs=-1,
                seed=50
            )

    def train(self, data_raw, target_raw, progress):

        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])
        data = np.reshape(data, (-1, num_input * self.timesteps))
        
        progress.emit(0)

        self.model.fit(data, target)
        predict, gt = list(), list()

        if self.mode == 1:
            predict = self.model.predict(data) 
            gt = target

        elif self.mode == 2:
            
            for i in range(0, len(data), 8):
                logits = None
                if i + 10 > len(data):
                    break
                for j in range(11):
                    x, y = data[i + j], target[i + j]
                    if logits != None:
                        x[-1] = float(logits[0])
                    x = np.reshape(x, (1, self.timesteps * num_input))
                    logits = self.model.predict(x)
                    if j > 2:
                        predict.append(logits[0])
                        gt.append(y)
                
                progress.emit(round((i + 1) / (len(data) + 1)))

        progress.emit(100)
        predict = np.array(predict, dtype=np.float32)
        target = np.array(gt, dtype=np.float32)
        loss = round(math.sqrt(metrics.mean_squared_error(target, predict)) * 100., 2)
        pickle.dump(self.model, open(os.path.join(output_path, f"xgboost_{self.mode}.pickle.dat"), "wb"))

        return loss
        
    def test(self, data_raw, target_raw, progress):
        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])

        model_path = os.path.join(output_path, f"xgboost_{self.mode}.pickle.dat")
        if not os.path.exists(model_path):
            return None, None, None

        progress.emit(0)
        self.model = pickle.load(open(model_path, "rb"))
        data = np.reshape(data, (-1, num_input * self.timesteps))
        predict, gt = list(), list()

        if self.mode == 1:
            predict = self.model.predict(data) 
            gt = target

        elif self.mode == 2:
            
            for i in range(0, len(data), 8):
                logits = None
                if i + 10 > len(data):
                    break
                for j in range(11):
                    x, y = data[i + j], target[i + j]
                    if logits != None:
                        x[-1] = float(logits[0])
                    x = np.reshape(x, (1, self.timesteps * num_input))
                    logits = self.model.predict(x)
                    if j > 2:
                        predict.append(logits[0])
                        gt.append(y)
                
                progress.emit(round((i + 1) / (len(data) + 1)))

        progress.emit(100)
        predict = np.array(predict, dtype=np.float32)
        target = np.array(gt, dtype=np.float32)
        loss = round(math.sqrt(metrics.mean_squared_error(target, predict)) * 100., 2)
        
        return predict, target, loss