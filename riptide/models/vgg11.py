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
        self.scalu = nn.Scalu()
        self.softmax = nn.Activation('softmax')

    def call(self, inputs, training=None, debug=False):
        layers = []
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
            layers.append(x)
            x = self.bn1(x, training=training)
            layers.append(x)
            x = self.pool1(x)
            layers.append(x)
        # When running in binary, need to reduce spread of normal distribution
        x = self.scale(x)
        layers.append(x)
        # Continue with binary layers.
        x = self.conv2(x)
        layers.append(x)
        x = self.bn2(
            x, conv_weights=self.conv2.weights[0].value(), training=training)
        layers.append(x)
        x = self.pool2(x)
        if self.scope.use_maxpool:
            layers.append(x)
        x = self.conv3(x)
        layers.append(x)
        x = self.bn3(
            x, conv_weights=self.conv3.weights[0].value(), training=training)
        layers.append(x)
        x = self.conv4(x)
        layers.append(x)
        x = self.bn4(
            x, conv_weights=self.conv4.weights[0].value(), training=training)
        layers.append(x)
        x = self.pool3(x)
        if self.scope.use_maxpool:
            layers.append(x)
        x = self.conv5(x)
        layers.append(x)
        x = self.bn5(
            x, conv_weights=self.conv5.weights[0].value(), training=training)
        layers.append(x)
        x = self.conv6(x)
        layers.append(x)
        x = self.bn6(
            x, conv_weights=self.conv6.weights[0].value(), training=training)
        layers.append(x)
        x = self.pool4(x)
        if self.scope.use_maxpool:
            layers.append(x)
        x = self.conv7(x)
        layers.append(x)
        x = self.bn7(
            x, conv_weights=self.conv7.weights[0].value(), training=training)
        layers.append(x)
        x = self.conv8(x)
        layers.append(x)
        x = self.bn8(
            x, conv_weights=self.conv8.weights[0].value(), training=training)
        layers.append(x)
        x = self.pool5(x)
        if self.scope.use_maxpool:
            layers.append(x)
        x = self.avgpool(x)
        layers.append(x)
        #with tf.name_scope('unbinarized'):
        x = self.classifier(x)
        layers.append(x)
        x = self.scalu(x)
        layers.append(x)
        x = self.softmax(x)
        layers.append(x)
        tf.compat.v1.summary.histogram('output', x)

        if debug:
            return layers
        else:
            return x
