import os
import tensorflow as tf
from riptide.binary import binary_layers as nn


class resnet18(tf.keras.Model):
    def __init__(self, classes=1000):
        super(resnet18, self).__init__()

        # Input Layer
        self.conv1 = nn.NormalConv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False)
        self.pool1 = nn.NormalMaxPool2D(pool_size=3, strides=2, padding='same')
        self.bn1 = nn.NormalBatchNormalization(center=False, scale=False)
        self.quantize = nn.EnterInteger(1.0)

        # BasicBlock 1
        self.block1_conv1 = nn.BinaryConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block1_bn1 = nn.NormalBatchNormalization()
        self.block1_conv2 = nn.BinaryConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block1_bn2 = nn.NormalBatchNormalization()
        self.block1_res = nn.ResidualConnect()

        # BasicBlock 2
        self.block2_conv1 = nn.BinaryConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block2_bn1 = nn.NormalBatchNormalization()
        self.block2_conv2 = nn.BinaryConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block2_bn2 = nn.NormalBatchNormalization()
        self.block2_res = nn.ResidualConnect()

        # BasicBlock 3
        self.block3_conv1 = nn.BinaryConv2D(
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False)
        self.block3_bn1 = nn.NormalBatchNormalization()
        self.block3_conv2 = nn.BinaryConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block3_bn2 = nn.NormalBatchNormalization()
        self.block3_down_conv = nn.BinaryConv2D(
            filters=128,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False)
        self.block3_down_bn = nn.NormalBatchNormalization()
        self.block3_res = nn.ResidualConnect()

        # BasicBlock 4
        self.block4_conv1 = nn.BinaryConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block4_bn1 = nn.NormalBatchNormalization()
        self.block4_conv2 = nn.BinaryConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block4_bn2 = nn.NormalBatchNormalization()
        self.block4_res = nn.ResidualConnect()

        # BasicBlock 5
        self.block5_conv1 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False)
        self.block5_bn1 = nn.NormalBatchNormalization()
        self.block5_conv2 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block5_bn2 = nn.NormalBatchNormalization()
        self.block5_down_conv = nn.BinaryConv2D(
            filters=256,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False)
        self.block5_down_bn = nn.NormalBatchNormalization()
        self.block5_res = nn.ResidualConnect()

        # BasicBlock 6
        self.block6_conv1 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block6_bn1 = nn.NormalBatchNormalization()
        self.block6_conv2 = nn.BinaryConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block6_bn2 = nn.NormalBatchNormalization()
        self.block6_res = nn.ResidualConnect()

        # BasicBlock 7
        self.block7_conv1 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False)
        self.block7_bn1 = nn.NormalBatchNormalization()
        self.block7_conv2 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block7_bn2 = nn.NormalBatchNormalization()
        self.block7_down_conv = nn.BinaryConv2D(
            filters=512,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False)
        self.block7_down_bn = nn.NormalBatchNormalization()
        self.block7_res = nn.ResidualConnect()

        # BasicBlock 8
        self.block8_conv1 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block8_bn1 = nn.NormalBatchNormalization()
        self.block8_conv2 = nn.BinaryConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False)
        self.block8_bn2 = nn.NormalBatchNormalization()
        self.block8_res = nn.ResidualConnect()

        self.avg_pool = nn.GlobalAveragePooling2D()
        self.flatten = nn.Flatten()
        self.dense = nn.BinaryDense(classes, use_bias=False)
        self.scalu = nn.Scalu()

    def call(self, inputs, training=None):
        with tf.name_scope('unbinarized'):
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.bn1(x, training=training)
            x = self.quantize(x)

        # Block 1
        residual = x
        x = self.block1_conv1(x)
        x = self.block1_bn1(
            x,
            conv_weights=self.block1_conv1.weights[0].value(),
            training=training)
        x = self.block1_conv2(x)
        x = self.block1_bn2(
            x,
            conv_weights=self.block1_conv2.weights[0].value(),
            training=training)
        next_res = x
        x = self.block1_res([x, residual])

        # Block 2
        residual = next_res
        x = self.block2_conv1(x)
        x = self.block2_bn1(
            x,
            conv_weights=self.block2_conv1.weights[0].value(),
            training=training)
        x = self.block2_conv2(x)
        x = self.block2_bn2(
            x,
            conv_weights=self.block2_conv2.weights[0].value(),
            training=training)
        next_res = x
        x = self.block2_res([x, residual])

        # Block 3
        residual = next_res
        x = self.block3_conv1(x)
        x = self.block3_bn1(
            x,
            conv_weights=self.block3_conv1.weights[0].value(),
            training=training)
        x = self.block3_conv2(x)
        x = self.block3_bn2(
            x,
            conv_weights=self.block3_conv2.weights[0].value(),
            training=training)
        next_res = x
        residual = self.block3_down_conv(residual)
        residual = self.block3_down_bn(
            residual,
            conv_weights=self.block3_down_conv.weights[0].value(),
            training=training)
        x = self.block3_res([x, residual])

        # Block 4
        residual = next_res
        x = self.block4_conv1(x)
        x = self.block4_bn1(
            x,
            conv_weights=self.block4_conv1.weights[0].value(),
            training=training)
        x = self.block4_conv2(x)
        x = self.block4_bn2(
            x,
            conv_weights=self.block4_conv2.weights[0].value(),
            training=training)
        next_res = x
        x = self.block4_res([x, residual])

        # Block 5
        residual = next_res
        x = self.block5_conv1(x)
        x = self.block5_bn1(
            x,
            conv_weights=self.block5_conv1.weights[0].value(),
            training=training)
        x = self.block5_conv2(x)
        x = self.block5_bn2(
            x,
            conv_weights=self.block5_conv2.weights[0].value(),
            training=training)
        next_res = x
        residual = self.block5_down_conv(residual)
        residual = self.block5_down_bn(
            residual,
            conv_weights=self.block5_down_conv.weights[0].value(),
            training=training)
        x = self.block5_res([x, residual])

        # Block 6
        residual = next_res
        x = self.block6_conv1(x)
        x = self.block6_bn1(
            x,
            conv_weights=self.block6_conv1.weights[0].value(),
            training=training)
        x = self.block6_conv2(x)
        x = self.block6_bn2(
            x,
            conv_weights=self.block6_conv2.weights[0].value(),
            training=training)
        next_res = x
        x = self.block6_res([x, residual])

        # Block 7
        residual = next_res
        x = self.block7_conv1(x)
        x = self.block7_bn1(
            x,
            conv_weights=self.block7_conv1.weights[0].value(),
            training=training)
        x = self.block7_conv2(x)
        x = self.block7_bn2(
            x,
            conv_weights=self.block7_conv2.weights[0].value(),
            training=training)
        next_res = x
        residual = self.block7_down_conv(residual)
        residual = self.block7_down_bn(
            residual,
            conv_weights=self.block7_down_conv.weights[0].value(),
            training=training)
        x = self.block7_res([x, residual])

        # Block 8
        residual = next_res
        x = self.block8_conv1(x)
        x = self.block8_bn1(
            x,
            conv_weights=self.block8_conv1.weights[0].value(),
            training=training)
        x = self.block8_conv2(x)
        x = self.block8_bn2(
            x,
            conv_weights=self.block8_conv2.weights[0].value(),
            training=training)
        x = self.block8_res([x, residual])

        # Output layers
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.scalu(x)

        return x
