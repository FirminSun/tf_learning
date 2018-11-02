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
# @Time    : 11/2/2018 2:45 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.datasets import mnist
import numpy as np

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

# 2. parameters
batch_size = 128
epochs = 2
num_classes = 10

# 3. train data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normalization
x_train  = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.

# reshape
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
# y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
# y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()


print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

# 4. build model
def build_model(inputs, num_classes):
    input_layer = tf.keras.Input(inputs)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',name='predictions')(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    return model

if __name__ == '__main__':
    inputs = (x_train.shape[1:])
    model = build_model(inputs, num_classes)
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.1), loss='categorical_crossentropy',metrics=['accuracy'])

    model.summary()

    history = model.fit(x_train, y_train_ohe, batch_size=batch_size,epochs=epochs, validation_data=(x_test, y_test_ohe), verbose=2)

    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=2)
    print('Final test loss and accuracy:', scores)
    saver = tfe.Saver(model.variables)
    saver.save('weights/k_02_logistic_regression/wegihts.ckpt')