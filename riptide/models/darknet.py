import os
import tensorflow as tf
from riptide.binary import binary_layers as nn
#from tensorflow.keras.models import Sequential
from riptide.utils.sequential import forward_layer_list


class DarkNet(tf.keras.Model):
    def __init__(self):
        super(DarkNet, self).__init__()

        self.conv1 = nn.NormalConv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn1 = nn.BatchNormalization()
        self.mxp1 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv2 = nn.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn2 = nn.BatchNormalization()
        self.mxp2 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv3 = nn.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn3 = nn.BatchNormalization()
        self.mxp3 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv4 = nn.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn4 = nn.BatchNormalization()
        self.mxp4 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv5 = nn.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn5 = nn.BatchNormalization()
        self.mxp5 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv6 = nn.Conv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn6 = nn.BatchNormalization()
        self.mxp6 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv7 = nn.Conv2D(
            filters=1024,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn7 = nn.BatchNormalization()
        self.avgpool = nn.GlobalAveragePooling2D()

        self.output_layer = nn.NormalDense(1000, use_bias=False)

    def call(self, inputs, training=None):
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.mxp1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.mxp2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.mxp3(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.mxp4(x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.mxp5(x)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.mxp6(x)
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.avgpool(x)
        with tf.name_scope('unbinarized'):
            x = self.output_layer(x)

        return x
