#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 11/2/2018 10:24 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import boston_housing
import numpy as np

# 1. GPU setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    return tf.Session(config=cfg)

get_session()
tfe.enable_eager_execution()
tfe.executing_eagerly()        # => True
tf.set_random_seed(0)
np.random.seed(0)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

# 2. parameters
batch_size = 128
epochs = 100

# 3. train data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.1)

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - std) / (std + 1e-8)

print('x train', x_train.shape, x_train.mean(), x_train.std())
print('y train', y_train.shape, y_train.mean(), y_train.std())
print('x test', x_test.shape, x_test.mean(), x_test.std())
print('y test', y_test.shape, y_test.mean(), y_test.std())

# 4. model
def build_model(input_shape=None):
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# 5 main
if __name__ == '__main__':
    input_shape = (x_train.shape[1:])
    model = build_model(input_shape)
    model.compile(optimizer = tf.train.GradientDescentOptimizer(0.05), loss='mse')

    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test))

    scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
    print('Test MSE:', scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/k_01_linear_regression/weights.ckpt')

    # I don't know how to restore it by now.
    # I try many methods, but failed.
