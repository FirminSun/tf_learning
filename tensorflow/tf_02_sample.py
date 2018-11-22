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
# @Time    : 11/21/2018 10:00 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf
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

if __name__ == '__main__':
    x = np.zeros((2,4))
    x[0,2:] = 1
    x[0,0:2] = 0.5
    var_x = tfe.Variable(x)

    a = tf.not_equal(var_x, 0.5)

    y = np.zeros((2,4))
    y[0,0]= 1
    y[0,1]= 2
    y[0,2]= 3
    y[0,3]= 4
    indices = tf.where(a)
    print(var_x)
    print(a)

    y_a = tf.gather_nd(y, indices)
    print(indices)
    print(y_a)

    label = tf.fill((tf.gather((10,4), [0])), -1)
    label = tf.boolean_mask(label,[True,True,True,False,True,True,False,True,True,False])

    x = tf.zeros(tf.shape(label))
    y = tf.to_float(label)


    print(label)