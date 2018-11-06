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
from tensorflow.python.keras.datasets import fashion_mnist
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

image_size = 28
batch_size = 32
epochs = 8
num_classes = 10

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((-1, image_size, image_size, 1))
x_test = x_test.reshape((-1, image_size, image_size, 1))

y_train_ohe = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=num_classes).numpy()

print('x train', x_train.shape)
print('y train', y_train_ohe.shape)
print('x test', x_test.shape)
print('y test', y_test_ohe.shape)

# 3x3 convolution
def conv3x3(channels, kernel_size=(3, 3), strides=(1, 1)):
    return tf.keras.layers.Conv2D(channels, kernel_size, strides=strides, padding='same', use_bias=False,
                                  kernel_initializer=tf.variance_scaling_initializer())

class ResnetBlock(tf.keras.Model):
    def __init__(self, channels, strides=(1, 1), residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(self.channels,strides=self.strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(self.channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(self.channels, kernel_size=(1,1), strides=self.strides)
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        residual = inputs
        x = self.bn1(inputs, training=training)
        x = tf.keras.activations.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.keras.activations.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual

        return x
class ResNet(tf.keras.Model):
    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.block_list = block_list
        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(channels=self.out_channels)

        self.blocks = []

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if layer_id == 0:
                    # only layer_id == 0, create the Identity Mapping
                    # you can do as you want, it's up to you.
                    block = ResnetBlock(self.out_channels, strides=(2, 2), residual_path=True)
                else:
                    block = ResnetBlock(self.out_channels, residual_path=False)

                # "register" this block to this model ; Without this, weights wont update.
                key = 'block_%d_%d' % (block_id + 1, layer_id + 1)
                setattr(self, key, block)

                self.blocks.append(block)

            self.out_channels *= 2

        self.final_bn = tf.keras.layers.BatchNormalization()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        out = self.conv_initial(inputs)

        # forward pass through all the blocks
        # build all the blocks
        for block in self.blocks:
            out = block(out, training=training)

        out = self.final_bn(out)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        output = tf.keras.activations.softmax(out)

        return output
if __name__ == '__main__':

    # build model and optimizer
    model = ResNet([2, 2], num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    dummy_x = tf.zeros((1, image_size, image_size, 1))
    model._set_inputs(dummy_x)

    print("Number of variables in the model :", len(model.variables))
    # model.summary()

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    saver = tfe.Saver(model.variables)
    saver.save('weights/k_05_02_resnet/weights.ckpt')