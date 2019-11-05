import tensorflow as tf
import tensorflow.keras.layers as nn
from riptide.anneal.anneal_funcs import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from riptide.binary.binary_layers import Scalu


class resnet18(tf.keras.Model):
    def __init__(self, classes=1000):
        super(resnet18, self).__init__()

        # Input Layer
        self.conv1 = nn.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False)
        self.pool1 = nn.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.b1 = nn.BatchNormalization()
        self.p1 = PACT()
        

        # BasicBlock 1
        self.block1_conv1 = SAWBConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent = self.p1)
        self.block1_bn1 = nn.BatchNormalization()
        self.block1_p1 = PACT()
        
        self.block1_conv2 = SAWBConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block1_p1
            )

        self.block1_down_conv = SAWBConv2D(  # Not in dorefa? But part of resnet?
            filters=64,
            kernel_size=1,
            strides=1,
            padding='valid',
            activation=None,
            use_bias=False,
            parent=self.p1
            )
        self.block1_res = tf.keras.layers.Add()

        # BasicBlock 2
        self.block2_bn1 = nn.BatchNormalization()
        self.block2_p1 = PACT()
        self.block2_conv1 = SAWBConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block2_p1
            )

        self.block2_bn2 = nn.BatchNormalization()
        self.block2_p2 = PACT()
        self.block2_conv2 = SAWBConv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block2_p2)
        self.block2_res = tf.keras.layers.Add()

        # BasicBlock 3
        self.block3_bn1 = nn.BatchNormalization()
        self.block3_p1 = PACT()
        self.block3_conv1 = SAWBConv2D(
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False, 
            parent=self.block3_p1)
        self.block3_bn2 = nn.BatchNormalization()
        self.block3_p2 = PACT()
        self.block3_conv2 = SAWBConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block3_p2)

        self.block3_down_conv = SAWBConv2D(
            filters=128,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False,
            parent=self.block3_p1)
        self.block3_res = tf.keras.layers.Add()

        # BasicBlock 4
        self.block4_bn1 = nn.BatchNormalization()
        self.block4_p1 = PACT()
        self.block4_conv1 = SAWBConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block4_p1)
        self.block4_bn2 = nn.BatchNormalization()
        self.block4_p2 = PACT()
        self.block4_conv2 = SAWBConv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block4_p2)
        self.block4_res = tf.keras.layers.Add()

        # BasicBlock 5
        self.block5_bn1 = nn.BatchNormalization()
        self.block5_p1 = PACT()
        self.block5_conv1 = SAWBConv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block5_p1)
        self.block5_bn2 = nn.BatchNormalization()
        self.block5_p2 = PACT()
        self.block5_conv2 = SAWBConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block5_p2)
        self.block5_down_conv = SAWBConv2D(
            filters=256,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False,
            parent=self.block5_p1)
        self.block5_res = tf.keras.layers.Add()

        # BasicBlock 6
        self.block6_bn1 = nn.BatchNormalization()
        self.block6_p1 = PACT()
        self.block6_conv1 = SAWBConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block6_p1)
        self.block6_bn2 = nn.BatchNormalization()
        self.block6_p2 = PACT()
        self.block6_conv2 = SAWBConv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block6_p2)
        self.block6_res = tf.keras.layers.Add()

        # BasicBlock 7
        self.block7_bn1 = nn.BatchNormalization()
        self.block7_p1 = PACT()
        self.block7_conv1 = SAWBConv2D(
            filters=512,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block7_p1)
        self.block7_bn2 = nn.BatchNormalization()
        self.block7_p2 = PACT()

        self.block7_conv2 = SAWBConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block7_p2)

        self.block7_down_conv = SAWBConv2D(
            filters=512,
            kernel_size=1,
            strides=2,
            padding='valid',
            activation=None,
            use_bias=False,
            parent=self.block7_p1)
        self.block7_res = tf.keras.layers.Add()

        # BasicBlock 8
        self.block8_bn1 = nn.BatchNormalization()
        self.block8_p1 = PACT()

        self.block8_conv1 = SAWBConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block8_p1)
        self.block8_bn2 = nn.BatchNormalization()
        self.block8_p2 = PACT()

        self.block8_conv2 = SAWBConv2D(
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            parent=self.block8_p2)
        self.block8_res = tf.keras.layers.Add()

        self.bn2 = nn.BatchNormalization()
        self.avg_pool = nn.GlobalAveragePooling2D()
        self.flatten = nn.Flatten()
        # self.q = nn.EnterInteger(1.5) # Dorefa fp here
        # self.p2 = PACT()
        self.dense = nn.Dense(classes, use_bias=False)
        self.scalu = Scalu()
        self.softmax = nn.Activation('softmax')

    def call(self, inputs, training=None):
        # Input layer
        # this name scope just affects names of parameters...the a2w1 has it...
        # the a2w2 doesn't oops
        #with tf.name_scope('unbinarized'):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.b1(x, training=training)
        # Block 1
        x = self.p1(x)
            
        residual = x
        x = self.block1_conv1(x)
        #return x
        x = self.block1_bn1(x, training=training)
        x = self.block1_p1(x)
        x = self.block1_conv2(x)

        residual = self.block1_down_conv(residual)
        x = self.block1_res([x, residual])

        # Block 2
        residual = x

        x = self.block2_bn1(x)
        x = self.block2_p1(x)
        x = self.block2_conv1(x)
        x = self.block2_bn2(x, training=training)
        x = self.block2_p2(x)
        x = self.block2_conv2(x)
        x = self.block2_res([x, residual])

        # Block 3
        x = self.block3_bn1(x, training=training)
        x = self.block3_p1(x)
        residual = x
        x = self.block3_conv1(x)
        x = self.block3_bn2(x,training=training)
        x = self.block3_p2(x)
        x = self.block3_conv2(x)
        residual = self.block3_down_conv(residual)
        x = self.block3_res([x, residual])

        # Block 4
        residual = x
        x = self.block4_bn1(x, training=training)
        x = self.block4_p1(x)
        x = self.block4_conv1(x)
        x = self.block4_bn2(x, training=training)
        x = self.block4_p2(x)
        x = self.block4_conv2(x)

        x = self.block4_res([x, residual])

        # Block 5
        x = self.block5_bn1(x, training=training)
        x = self.block5_p1(x)
        residual = x
        x = self.block5_conv1(x)
        #return x
        x = self.block5_bn2(x, training=training)
        x = self.block5_p2(x)
        x = self.block5_conv2(x)
 
        residual = self.block5_down_conv(residual)
        x = self.block5_res([x, residual])

        # Block 6
        residual = x
        x = self.block6_bn1(x, training=training)
        x = self.block6_p1(x)
        x = self.block6_conv1(x)
        x = self.block6_bn2(x, training=training)
        x = self.block6_p2(x)
        x = self.block6_conv2(x)
        #return x
        x = self.block6_res([x, residual])

        # Block 7
        x = self.block7_bn1(x, training=training)
        x = self.block7_p1(x)
        residual = x
        x = self.block7_conv1(x)
        #return x
        x = self.block7_bn2(x, training=training)
        x = self.block7_p2(x)
        x = self.block7_conv2(x)
        #return x
        residual = self.block7_down_conv(residual)
        x = self.block7_res([x, residual])

        # Block 8
        residual = x
        x = self.block8_bn1(x, training=training)
        x = self.block8_p1(x)
        x = self.block8_conv1(x)
        x = self.block8_bn2(x, training=training)
        x = self.block8_p2(x)
        x = self.block8_conv2(x)
        x = self.block8_res([x, residual])

        # Output layers
        x = self.bn2(x, training=training)
        x = self.avg_pool(x)
        x = self.flatten(x)
        # x = self.q(x)
        # x = self.p2(x)
        x = self.dense(x)
        x = self.scalu(x) # only for 1-bit
        x = self.softmax(x)

        return x
