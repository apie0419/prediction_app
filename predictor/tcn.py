from .model import TCN
from .utils import process_data, rmse
from tensorflow.contrib.eager.python import tfe
from PyQt5.QtCore import pyqtSignal

import tensorflow as tf
import numpy as np
import os

tf.enable_eager_execution()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(BASE_PATH, "weights")

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

class tcn():

    def __init__(self):
        self.batch_size      = 16
        self.hidden_units    = 16
        self.dropout         = 0.3
        self.epochs          = 100
        self.ksize           = 3
        self.levels          = 5
        self.output_dim      = 1

        self.global_step   = tf.Variable(0, trainable=False)
        channel_sizes = [self.hidden_units] * self.levels
        self.learning_rate = tf.compat.v1.train.exponential_decay(0.0001, self.global_step, 3000, 0.7, staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.model = TCN(self.output_dim, channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def init_model(self, use_gpu=False, device=0, mode=1):
        self.mode = mode
        self.use_gpu = use_gpu
        self.device = device

        if self.mode == 1:
            self.timesteps = 24
        elif self.mode == 2:
            self.timesteps = 8

    def loss_function(self, batch_x, batch_y):
        logits = self.model(batch_x, training=True)
        
        return rmse(logits, batch_y)

    def train(self, data_raw, target_raw):
        
        # progress.emit(0)
        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])

        trainset = tf.data.Dataset.from_tensor_slices((data, target))
        if self.use_gpu:
            todevice = f"/gpu:{self.device}"
        else:
            todevice = f"/cpu:0"

        train_losses = list()

        with tf.device(todevice):
    
            for epoch in range(1, self.epochs + 1):
                train = trainset.batch(self.batch_size, drop_remainder=True)
                
                for batch_x, batch_y in tfe.Iterator(train):
                    batch_x = tf.reshape(batch_x, (self.batch_size, self.timesteps, num_input))
                    batch_x = tf.dtypes.cast(batch_x, tf.float32)
                    batch_y = tf.dtypes.cast(batch_y, tf.float32)
                    self.optimizer.minimize(lambda: self.loss_function(batch_x, batch_y), global_step=self.global_step)
                
                predict, gt = list(), list()
                if self.mode == 1:
                    for batch_x, batch_y in tfe.Iterator(train):
                        batch_x = tf.reshape(batch_x, (self.batch_size, self.timesteps, num_input))
                        batch_x = tf.dtypes.cast(batch_x, tf.float32)
                        batch_y = tf.dtypes.cast(batch_y, tf.float32)
                        logits = self.model(batch_x, training=False)
                        predict.extend(logits.numpy().flatten().tolist())
                        gt.extend(batch_y.numpy().flatten().tolist())

                elif self.mode == 2:
                    for i in range(0, len(data), 8):
                        logits = None
                        if i + 10 > len(data):
                            break
                        for j in range(11):
                            x, y = data[i + j], target[i + j]
                            if logits != None:
                                x[-1][-1] = float(logits.numpy()[0][0])
                            x = tf.convert_to_tensor(x, dtype=tf.float32)
                            x = tf.reshape(x, (1, self.timesteps, num_input))
                            y = tf.convert_to_tensor(y, dtype=tf.float32)
                            
                            logits = self.model(x, training=False)
                            if j > 2:
                                predict.append(logits.numpy()[0][0])
                                gt.append(y.numpy())

                predict = np.array(predict)
                gt = np.array(gt)
                train_losses.append(round(rmse(predict, gt).numpy() * 100., 2))
                # p = round(epoch / (self.epochs + 1) * 100.)
                # progress.emit(p)
                yield np.array(train_losses)
            
            # progress.emit(100)
            self.model.save_weights(os.path.join(output_path, f"tcn_weight_{self.mode}.ckpt"))

        return np.array(train_losses)
    
    def test(self, data_raw, target_raw, progress):
        model_path = os.path.join(output_path, f"tcn_weight_{self.mode}.ckpt")
        
        try:
            self.model.load_weights(os.path.join(output_path, f"tcn_weight_{self.mode}.ckpt"))
        except:
            return None, None, None

        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])
        predict, gt = list(), list()
        testset = tf.data.Dataset.from_tensor_slices((data, target))
        progress.emit(0)


        if self.use_gpu:
            todevice = f"/gpu:{self.device}"
        else:
            todevice = f"/cpu:0"

        with tf.device(todevice):
            if self.mode == 1:
                test = testset.batch(self.batch_size, drop_remainder=True)
                for batch_x, batch_y in tfe.Iterator(test):
                    batch_x = tf.reshape(batch_x, (self.batch_size, self.timesteps, num_input))
                    batch_x = tf.dtypes.cast(batch_x, tf.float32)
                    batch_y = tf.dtypes.cast(batch_y, tf.float32)
                    logits = self.model(batch_x, training=False)
                    predict.extend(logits.numpy().flatten().tolist())
                    gt.extend(batch_y.numpy().flatten().tolist())

            elif self.mode == 2:
                for i in range(0, len(data), 8):
                    logits = None
                    if i + 10 > len(data):
                        break
                    for j in range(11):
                        x, y = data[i + j], target[i + j]
                        if logits != None:
                            x[-1][-1] = float(logits.numpy()[0][0])
                        x = tf.convert_to_tensor(x, dtype=tf.float32)
                        x = tf.reshape(x, (1, self.timesteps, num_input))
                        y = tf.convert_to_tensor(y, dtype=tf.float32)
                        
                        logits = self.model(x, training=False)
                        if j > 2:
                            predict.append(logits.numpy()[0][0])
                            gt.append(y.numpy())
                    p = round(i / (len(data) - 1) * 100., 2)
                    progress.emit(p)

        progress.emit(100)
        test_predict = np.array(predict)
        test_target = np.array(gt)
        test_loss = round(rmse(test_predict, test_target).numpy() * 100., 2)
        return test_predict, test_target, test_loss
            