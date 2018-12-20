import os
import tensorflow as tf
from riptide.binary import binary_layers as nn
from riptide.utils.sequential import forward_layer_list


class vggnet(tf.keras.Model):
    def __init__(self, classes=1000):
        super(vggnet, self).__init__()

        self.conv1 = nn.NormalConv2D(
            filters=96,
            kernel_size=7,
            strides=2,
            padding='same',
            activation='relu',
            use_bias=False)
        self.pool1 = nn.NormalMaxPool2D(pool_size=2, strides=2)
        self.bn1 = nn.NormalBatchNormalization()
        self.quantize = nn.Scale(1.0)

        self.conv2 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn2 = nn.BatchNormalization()

        self.conv3 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn3 = nn.BatchNormalization()

        self.conv4 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.pool4 = nn.MaxPool2D(pool_size=2, strides=2)
        self.bn4 = nn.BatchNormalization()

        self.conv5 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn5 = nn.BatchNormalization()

        self.conv6 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn6 = nn.BatchNormalization()

        self.conv7 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.pool7 = nn.MaxPool2D(pool_size=2, strides=2)
        self.bn7 = nn.BatchNormalization()

        self.conv8 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn8 = nn.BatchNormalization()

        self.conv9 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn9 = nn.BatchNormalization()

        self.conv10 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.pool10 = nn.MaxPool2D(pool_size=2, strides=2)
        self.bn10 = nn.BatchNormalization()
        self.flatten = nn.Flatten()

        self.dense1 = nn.BinaryDense(4096, use_bias=False, activation='relu')
        self.bn11 = nn.BatchNormalization()
        self.dense2 = nn.BinaryDense(4096, use_bias=False, activation='relu')
        self.bn12 = nn.BatchNormalization()
        self.dense3 = nn.BinaryDense(classes, use_bias=False)
        self.scalu = nn.Scalu()

    def call(self, inputs, training=None):
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.bn1(x)
        x = self.quantize(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.pool7(x)
        x = self.bn7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.conv10(x)
        x = self.pool10(x)
        x = self.bn10(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn11(x)
        x = self.dense2(x)
        x = self.bn12(x)
        x = self.dense3(x)
        x = self.scalu(x)
        tf.summary.histogram('output', x)

        return x
