import os
import tensorflow as tf
from riptide.binary import binary_layers as nn
#from tensorflow.keras.models import Sequential
from riptide.utils.sequential import forward_layer_list


class vgg11(tf.keras.Model):
    def __init__(self, classes=1000):
        super(vgg11, self).__init__()

        # Set up configurable maxpool or stride dimension reduction.
        self.scope = nn.Config.current
        use_maxpool = self.scope.use_maxpool
        if use_maxpool:
            reduce_stride = 1
        else:
            reduce_stride = 2

        self.conv1 = nn.NormalConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu')
        self.bn1 = nn.NormalBatchNormalization()
        self.pool1 = nn.NormalMaxPool2D(pool_size=2, strides=2)
        self.scale = nn.Scale(0.5)

        self.conv2 = nn.BinaryConv2D(
            filters=128,
            kernel_size=3,
            strides=reduce_stride,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn2 = nn.BatchNormalization()
        self.pool2 = nn.MaxPool2D(pool_size=2, strides=2)

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
            strides=reduce_stride,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn4 = nn.BatchNormalization()
        self.pool3 = nn.MaxPool2D(pool_size=2, strides=2)

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
            strides=reduce_stride,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn6 = nn.BatchNormalization()
        self.pool4 = nn.MaxPool2D(pool_size=2, strides=2)

        self.conv7 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn7 = nn.BatchNormalization()
        self.conv8 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=reduce_stride,
            padding='same',
            activation='relu',
            use_bias=False)
        self.bn8 = nn.BatchNormalization()
        self.pool5 = nn.MaxPool2D(pool_size=2, strides=2)

        self.avgpool = nn.GlobalAveragePooling2D()
        self.classifier = nn.BinaryDense(classes, use_bias=False)

    def call(self, inputs, training=None):
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = self.pool1(x)
        # When running in binary, need to reduce spread of normal distribution
        x = self.scale(x)
        # Continue with binary layers.
        x = self.conv2(x)
        x = self.bn2(
            x, conv_weights=self.conv2.weights[0].value(), training=training)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(
            x, conv_weights=self.conv3.weights[0].value(), training=training)
        x = self.conv4(x)
        x = self.bn4(
            x, conv_weights=self.conv4.weights[0].value(), training=training)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.bn5(
            x, conv_weights=self.conv5.weights[0].value(), training=training)
        x = self.conv6(x)
        x = self.bn6(
            x, conv_weights=self.conv6.weights[0].value(), training=training)
        x = self.pool4(x)
        x = self.conv7(x)
        x = self.bn7(
            x, conv_weights=self.conv7.weights[0].value(), training=training)
        x = self.conv8(x)
        x = self.bn8(
            x, conv_weights=self.conv8.weights[0].value(), training=training)
        x = self.pool5(x)
        x = self.avgpool(x)
        #with tf.name_scope('unbinarized'):
        x = self.classifier(x)
        tf.summary.histogram('output', x)

        return x
