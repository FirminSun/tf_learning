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
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data


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

W = tf.get_variable(name="W", shape=(784, 10))
b = tf.get_variable(name="b", shape=(10, ))


def softmax_model(image_batch):
    model_output = tf.nn.softmax(tf.matmul(image_batch, W) + b)
    return model_output


def cross_entropy(model_output, label_batch):
    loss = tf.reduce_mean(
        -tf.reduce_sum(label_batch * tf.log(model_output),
        reduction_indices=[1]))
    return loss


@tfe.implicit_value_and_gradients
def cal_gradient(image_batch, label_batch):
    return cross_entropy(softmax_model(image_batch), label_batch)


if __name__ == '__main__':
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels))\
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
        .shuffle(buffer_size=1000)\
        .batch(64)\

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    epochs = 2
    for e in range(epochs):
        for step, (image_batch, label_batch) in enumerate(tfe.Iterator(train_ds)):
            loss, grads_and_vars = cal_gradient(image_batch, label_batch)
            optimizer.apply_gradients(grads_and_vars)
            print("step: {}  loss: {}".format(step, loss.numpy()))

        model_test_output = softmax_model(data.test.images)
        model_test_label = data.test.labels
        correct_prediction = tf.equal(tf.argmax(model_test_output, 1), tf.argmax(model_test_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("test accuracy = {}".format(accuracy.numpy()))
