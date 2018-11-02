# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 7/18/2018 3:45 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    return tf.Session(config=cfg)

get_session()
tf.enable_eager_execution()
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"


x = [[1],[1]]
y = [[2,3]]
print(x)
print(y)
m = tf.matmul(y,x)
print("{}".format(m))

a = tf.constant([[1,2],[3,4]])
b = tf.add(a, 1)

print(a)
print(b)

print(a * b)