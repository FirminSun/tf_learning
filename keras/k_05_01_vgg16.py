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
# @Time    : 11/5/2018 11:35 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf
from keras.datasets import cifar10
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

image_size = 32
batch_size = 128
epochs = 8
num_classes = 10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape((-1, image_size, image_size, 3))
x_test = x_test.reshape((-1, image_size, image_size, 3))

def normalize(x_train, x_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test

y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

def conv3x3(channels, kernel_size=(3, 3) , strides=(1, 1), name=''):
    conv = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_size, strides=strides,
                               padding='same', use_bias=False,name=name,
                               kernel_initializer=tf.variance_scaling_initializer())

    return conv

class VGG16(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        # Block 1
        self.block1_conv1 = conv3x3(64, name='block1_conv1')
        self.block1_conv2 = conv3x3(64, name='block1_conv2')
        self.block1_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        self.block2_conv1 = conv3x3(128, name='block2_conv1')
        self.block2_conv2 = conv3x3(128, name='block2_conv2')
        self.block2_conv3 = conv3x3(128, name='block2_conv3')
        self.block2_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.block3_conv1 = conv3x3(256, name='block3_conv1')
        self.block3_conv2 = conv3x3(256, name='block3_conv2')
        self.block3_conv3 = conv3x3(256, name='block3_conv3')
        self.block3_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        # Block 4
        self.block4_conv1 = conv3x3(512, name='block4_conv1')
        self.block4_conv2 = conv3x3(512, name='block4_conv2')
        self.block4_conv3 = conv3x3(512, name='block4_conv3')
        self.block4_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        # Block 5
        self.block5_conv1 = conv3x3(512, name='block5_conv1')
        self.block5_conv2 = conv3x3(512, name='block5_conv2')
        self.block5_conv3 = conv3x3(512, name='block5_conv3')
        self.block5_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense1 = tf.keras.layers.Dense(units=4096,name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=4096, name='dense2')
        self.classifier = tf.keras.layers.Dense(num_classes ,name='classifier')

    def call(self, inputs, training=None, mask=None):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = tf.keras.activations.relu(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.block3_pool(x)

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = tf.keras.activations.relu(x)
        x = self.block5_pool(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.keras.activations.relu(x)

        x = self.dense2(x)
        x = tf.keras.activations.relu(x)

        output = self.classifier(x)
        output = tf.keras.activations.softmax(output)

        return output


if __name__ == '__main__':
    model = VGG16(num_classes)

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.metrics_names = ['loss']

    dummy_x = tf.zeros((1, image_size, image_size, 3))
    model._set_inputs(dummy_x)
    model.summary()

    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,validation_data=(x_test, y_test_ohe),verbose=2)

    scores = model.evaluate(x_test, y_test_ohe, batch_size=batch_size, verbose=1)
    print('Final test loss and accuracy :', scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/k_05_01_vgg16/weights.ckpt')