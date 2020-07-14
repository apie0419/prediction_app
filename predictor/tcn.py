from .model import TCN
from .utils import process_data, rmse
from tensorflow.contrib.eager.python import tfe
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
        self.GPU           = 6
        self.batch_size    = 16
        self.hidden_units  = 16
        self.dropout       = 0.3
        self.epochs        = 100
        self.ksize         = 3
        self.levels        = 5
        self.output_dim    = 1
        self.timesteps     = 8

        global_step   = tf.Variable(0, trainable=False)
        channel_sizes = [self.hidden_units] * self.levels
        self.learning_rate = tf.compat.v1.train.exponential_decay(0.0001, global_step, 3000, 0.7, staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.model = TCN(self.output_dim, channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def loss_function(self, batch_x, batch_y):
        logits = self.model(batch_x, training=True)
        
        return rmse(logits, batch_y)

    def train(self, data_raw, target_raw):
        """
        Todo:
        1. Progress Bar
        2. XGBOOST
        3. hourahead train & test
        """
        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])

        trainset = tf.data.Dataset.from_tensor_slices((data, target))
        with tf.device(f"/gpu:{self.GPU}"):
    
            for epoch in range(1, self.epochs + 1):
                train = trainset.batch(self.batch_size, drop_remainder=True)
                
                for batch_x, batch_y in tfe.Iterator(train):
                    batch_x = tf.reshape(batch_x, (self.batch_size, self.timesteps, num_input))
                    batch_x = tf.dtypes.cast(batch_x, tf.float32)
                    batch_y = tf.dtypes.cast(batch_y, tf.float32)
                    self.optimizer.minimize(lambda: self.loss_function(batch_x, batch_y), global_step=self.global_step)
                
                predict, target = list(), list()
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
                            target.append(y.numpy())

                predict = np.array(predict)
                target = np.array(target)
                train_loss = round(rmse(predict, target) * 100., 2)

            model.save_weights(os.path.join(output_path, "tcn_weight.ckpt"))
    
    def test(self, data_raw, target_raw, layout):
        
        self.model.load_weights(os.path.join(output_path, "tcn_weight.ckpt"))
        data, target = process_data(self.timesteps, data_raw, target_raw)
        num_input = len(data[0][0])
        predict, gt = list(), list()

        with tf.device("/cpu:0"):
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
                progress = round(i / (len(data) - 1) * 100., 2)

        test_predict = np.array(predict)
        test_target = np.array(gt)
        test_loss = round(rmse(test_predict, test_target).numpy() * 100., 2)
        return test_predict, test_target, test_loss
            