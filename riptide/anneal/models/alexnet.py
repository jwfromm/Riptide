import tensorflow as tf
import tensorflow.keras.layers as nn
from riptide.anneal.anneal_funcs import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential


class alexnet(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(alexnet, self).__init__(*args, **kwargs)
        l2_reg = 5e-6

        # Layer 1
        self.c1 = nn.Conv2D(
            96, (11, 11),
            strides=(4, 4),
            padding='same',
            kernel_regularizer=l2(l2_reg))
        self.b1 = nn.BatchNormalization()
        self.p1 = PACT()
        self.m1 = nn.MaxPooling2D(pool_size=(2, 2))

        # Layer 2
        self.c2 = SAWBConv2D(
            256, (5, 5), padding='same', kernel_regularizer=l2(l2_reg))
        self.b2 = nn.BatchNormalization()
        self.p2 = PACT()
        self.m2 = nn.MaxPooling2D(pool_size=(2, 2))

        # Layer 3
        self.c3 = SAWBConv2D(
            384, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))
        self.b3 = nn.BatchNormalization()
        self.p3 = PACT()

        # Layer 4
        self.c4 = SAWBConv2D(
            384, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))
        self.b4 = nn.BatchNormalization()
        self.p4 = PACT()

        # Layer 5
        self.c5 = SAWBConv2D(
            256, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))
        self.b5 = nn.BatchNormalization()
        self.p5 = PACT()
        self.m5 = nn.MaxPooling2D()

        # FC
        self.f6 = nn.Flatten()
        self.d6 = SAWBDense(4096, kernel_regularizer=l2(l2_reg))
        self.b6 = nn.BatchNormalization()
        self.p6 = PACT()
        self.do6 = nn.Dropout(0.5)

        # FC2
        self.d7 = SAWBDense(4096, kernel_regularizer=l2(l2_reg))
        self.b7 = nn.BatchNormalization()
        self.p7 = nn.Activation('relu')
        self.do7 = nn.Dropout(0.5)

        # Output
        self.d8 = nn.Dense(1000, kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training=True):
        x = self.c1(inputs)
        x = self.b1(x, training=training)
        x = self.p1(x)
        x = self.m1(x)

        x = self.c2(x)
        x = self.b2(x, training=training)
        x = self.p2(x)
        x = self.m2(x)

        x = self.c3(x)
        x = self.b3(x, training=training)
        x = self.p3(x)

        x = self.c4(x)
        x = self.b4(x, training=training)
        x = self.p4(x)

        x = self.c5(x)
        x = self.b5(x, training=training)
        x = self.p5(x)
        x = self.m5(x)

        x = self.f6(x)
        x = self.d6(x)
        x = self.b6(x, training=training)
        x = self.p6(x)
        x = self.do6(x, training=training)

        x = self.d7(x)
        x = self.b7(x, training=training)
        x = self.p7(x)
        x = self.do7(x, training=training)

        x = self.d8(x)

        return x
