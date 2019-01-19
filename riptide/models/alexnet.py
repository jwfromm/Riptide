import os
import tensorflow as tf
from riptide.binary import binary_layers as nn

class alexnet(tf.keras.Model):
    def __init__(self, classes=1000):
        super(alexnet, self).__init__()
        self.conv1 = nn.NormalConv2D(
                filters=64,
                kernel_size=11,
                strides=4,
                padding='same',
                activation=None,
                use_bias=False)
        self.pool1 = nn.NormalMaxPool2D(pool_size=2, strides=2)
        self.bn1 = nn.NormalBatchNormalization(center=False, scale=False)
        self.quantize = nn.Scale(1.0)

        self.conv2 = nn.BinaryConv2D(
                filters=192,
                kernel_size=5,
                strides=1,
                padding='same',
                activation=None,
                use_bias=False)
        self.pool2 = nn.MaxPool2D(pool_size=2, strides=2)
        self.bn2 = nn.BatchNormalization()

        self.conv3 = nn.BinaryConv2D(
                filters=384,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=None,
                use_bias=False)
        self.bn3 = nn.BatchNormalization()

        self.conv4 = nn.BinaryConv2D(
                filters=384,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=None,
                use_bias=False)
        self.bn4 = nn.BatchNormalization()

        self.conv5 = nn.BinaryConv2D(
                filters=256,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=None,
                use_bias=False)
        self.pool5 = nn.MaxPool2D(pool_size=2, strides=2)
        self.bn5 = nn.BatchNormalization()

        self.flatten = nn.Flatten()
        self.dense6 = nn.BinaryDense(4096, use_bias=False, activation=None)
        self.bn6 = nn.BatchNormalization(binary_dense=True)
        self.dense7 = nn.BinaryDense(4096, use_bias=False, activation=None)
        self.bn7 = nn.BatchNormalization(binary_dense=True)
        self.dense8 = nn.BinaryDense(classes, use_bias=False, activation=None)
        self.scalu = nn.Scalu()

    def call(self, inputs, training=None):
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.bn1(x)
            x = self.quantize(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, conv_weights = self.conv2.weights[0].value(), training=training)

        x = self.conv3(x)
        x = self.bn3(x, conv_weights = self.conv3.weights[0].value(), training=training)

        x = self.conv4(x)
        x = self.bn4(x, conv_weights = self.conv4.weights[0].value(), training=training)

        x = self.conv5(x)
        x = self.pool5(x)
        x = self.bn5(x, conv_weights = self.conv5.weights[0].value(), training=training)

        x = self.flatten(x)
        x = self.dense6(x)
        x = self.bn6(x, conv_weights=self.dense6.weights[0].value(), training=training)
        x = self.dense7(x)
        x = self.bn7(x, conv_weights=self.dense7.weights[0].value(), training=training)
        x = self.dense8(x)
        x = self.scalu(x)

        tf.summary.histogram('output', x)

        return x
