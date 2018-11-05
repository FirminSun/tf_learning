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
# @Time    : 11/5/2018 11:15 AM
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

batch_size = 128
epochs = 8
num_classes = 10

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

class ConvBnReluBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBnReluBlock, self).__init__()
        self.cnn = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer='random_normal')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.cnn(inputs)
        x = tf.keras.activations.relu(x)
        return x

class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.block1 = ConvBnReluBlock(filters=16, kernel_size=(5, 5), strides=(2, 2))
        self.block2 = ConvBnReluBlock(filters=32, kernel_size=(5, 5), strides=(2, 2))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def  call(self, inputs, training=None, mask=None):
        x = self.block1(inputs)
        x = self.block2(x)

        x = self.pool(x)
        output = self.classifier(x)
        output = tf.keras.activations.softmax(output)

        return output

if __name__ == '__main__':
    model = CNN(num_classes)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_ohe), verbose=1)

    scores = model.evaluate(x_test, y_test_ohe, batch_size=batch_size, verbose=1)
    print('Final test loss and accuracy :', scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/k_04_02_cnn_block/weights.ckpt')