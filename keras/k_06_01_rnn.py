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
# @Time    : 11/6/2018 5:39 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.contrib.eager.python import tfe

# 1. GPU setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# constants
units = 128
batch_size = 100
epochs = 2
num_classes = 10

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

class RNN(tf.keras.Model):
    def __init__(self, units, num_classes, num_layers=1):
        super(RNN, self).__init__()
        self.cells = [tf.keras.layers.LSTMCell(units=units, implementation=2) for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(self.cells, unroll=True)
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.rnn(inputs)

        out = self.classifier(x)
        # output = tf.nn.softmax(out)
        output = tf.keras.activations.softmax(out)
        return  output


if __name__ == '__main__':
    model = RNN(units=units, num_classes=num_classes, num_layers=2)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    dummy_x = tf.zeros((1, 28, 28))
    model._set_inputs(dummy_x)

    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_ohe), verbose=1)

    scores = model.evaluate(x_test, y_test_ohe, batch_size=batch_size, verbose=1)
    print('Final test loss and accuracy :', scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/k_06_01_rnn/weights.ckpt')