import os
import tensorflow as tf
from riptide.binary import binary_layers as nn
#from tensorflow.keras.models import Sequential
from riptide.utils.sequential import forward_layer_list


class CifarNet(tf.keras.Model):
    def __init__(self):
        super(CifarNet, self).__init__()

        self.conv1 = nn.NormalConv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn1 = nn.BatchNormalization()

        self.conv2 = nn.Conv2D(
            filters=16,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn2 = nn.BatchNormalization()

        self.conv3 = nn.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn3 = nn.BatchNormalization()

        self.conv4 = nn.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn4 = nn.BatchNormalization()

        self.conv5 = nn.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn5 = nn.BatchNormalization()

        self.conv6 = nn.Conv2D(
            filters=64,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn6 = nn.BatchNormalization()

        self.conv7 = nn.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn7 = nn.BatchNormalization()

        self.global_pooling = nn.GlobalAveragePooling2D()

        self.dense = nn.NormalDense(10, use_bias=False)

    def call(self, inputs, training=None):
        with tf.name_scope('normal'):
            x = self.conv1(inputs)
        tf.summary.histogram('first_layer_weights', self.conv1.kernel)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        tf.summary.histogram('bn4_input', x)
        x = self.bn4(x, training=training)
        tf.summary.histogram('bn4_output', x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.global_pooling(x)
        with tf.name_scope('normal'):
            x = self.dense(x)
        tf.summary.histogram('output', x)

        return x
