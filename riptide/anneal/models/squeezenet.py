import tensorflow as tf
import tensorflow.keras.layers as nn
from riptide.anneal.anneal_funcs import *
from tensorflow.keras.regularizers import l2

class SqueezeNet(tf.keras.models.Model):
    def __init__(self, classes=1000):
        super(SqueezeNet, self).__init__()
        self.classes = classes
        l2_reg = 5e-6

        self.c0 = nn.Conv2D(kernel_size=7, strides=2, filters=96, padding='same', activation='relu', kernel_regularizer=l2(l2_reg))
        self.mp0 = nn.MaxPooling2D(pool_size=2)
        self.b0 = nn.BatchNormalization()
        self.p0 = PACT()

        # Fire 1
        self.f1c1 = SAWBConv2D(filters=32, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f1b1 = nn.BatchNormalization()
        self.f1p1 = PACT()

        self.f1c2 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f1b2 = nn.BatchNormalization()
        self.f1p2 = PACT()

        self.f1c3 = SAWBConv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f1b3 = nn.BatchNormalization()
        self.f1p3 = PACT()

        self.f1concat = nn.Concatenate(axis=-1)

        # Fire 2
        self.f2c1 = SAWBConv2D(filters=32, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f2b1 = nn.BatchNormalization()
        self.f2p1 = PACT()

        self.f2c2 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f2b2 = nn.BatchNormalization()
        self.f2p2 = PACT()

        self.f2c3 = SAWBConv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f2b3 = nn.BatchNormalization()
        self.f2p3 = PACT()

        self.f2concat = nn.Concatenate(axis=-1)

        # Fire 3
        self.f3c1 = SAWBConv2D(filters=32, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f3b1 = nn.BatchNormalization()
        self.f3p1 = PACT()

        self.f3c2 = SAWBConv2D(filters=128, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f3b2 = nn.BatchNormalization()
        self.f3p2 = PACT()

        self.f3c3 = SAWBConv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f3b3 = nn.BatchNormalization()
        self.f3p3 = PACT()

        self.f3concat = nn.Concatenate(axis=-1)

        self.mp3 = nn.MaxPooling2D(pool_size=2)

        # Fire 4
        self.f4c1 = SAWBConv2D(filters=32, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f4b1 = nn.BatchNormalization()
        self.f4p1 = PACT()

        self.f4c2 = SAWBConv2D(filters=128, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f4b2 = nn.BatchNormalization()
        self.f4p2 = PACT()

        self.f4c3 = SAWBConv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f4b3 = nn.BatchNormalization()
        self.f4p3 = PACT()

        self.f4concat = nn.Concatenate(axis=-1)

        # Fire 5
        self.f5c1 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f5b1 = nn.BatchNormalization()
        self.f5p1 = PACT()

        self.f5c2 = SAWBConv2D(filters=192, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f5b2 = nn.BatchNormalization()
        self.f5p2 = PACT()

        self.f5c3 = SAWBConv2D(filters=192, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f5b3 = nn.BatchNormalization()
        self.f5p3 = PACT()

        self.f5concat = nn.Concatenate(axis=-1)

        # Fire 6
        self.f6c1 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f6b1 = nn.BatchNormalization()
        self.f6p1 = PACT()

        self.f6c2 = SAWBConv2D(filters=192, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f6b2 = nn.BatchNormalization()
        self.f6p2 = PACT()

        self.f6c3 = SAWBConv2D(filters=192, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f6b3 = nn.BatchNormalization()
        self.f6p3 = PACT()

        self.f6concat = nn.Concatenate(axis=-1)

        # Fire 7
        self.f7c1 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f7b1 = nn.BatchNormalization()
        self.f7p1 = PACT()

        self.f7c2 = SAWBConv2D(filters=256, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f7b2 = nn.BatchNormalization()
        self.f7p2 = PACT()

        self.f7c3 = SAWBConv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f7b3 = nn.BatchNormalization()
        self.f7p3 = PACT()

        self.f7concat = nn.Concatenate(axis=-1)

        self.mp7 = nn.MaxPooling2D(pool_size=2)

        # Fire 8
        self.f8c1 = SAWBConv2D(filters=64, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f8b1 = nn.BatchNormalization()
        self.f8p1 = PACT()

        self.f8c2 = SAWBConv2D(filters=256, kernel_size=1, padding='same', kernel_regularizer=l2(l2_reg))
        self.f8b2 = nn.BatchNormalization()
        self.f8p2 = PACT()

        self.f8c3 = SAWBConv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(l2_reg))
        self.f8b3 = nn.BatchNormalization()
        self.f8p3 = PACT()

        self.f8concat = nn.Concatenate(axis=-1)

        # Output
        self.avgpool = nn.GlobalAveragePooling2D()
        self.classifier = nn.Dense(1000, activation='softmax')

    def call(self, x, training=None):
        y = self.c0(x)
        y = self.mp0(y)
        y = self.b0(y, training=training)
        y = self.p0(y)

        # Fire 1
        y = self.f1c1(y)
        y = self.f1b1(y, training=training)
        y = self.f1p1(y)
        y1x = self.f1c2(y)
        y1x = self.f1b2(y1x, training=training)
        y1x = self.f1p2(y1x)
        y3x = self.f1c3(y)
        y3x = self.f1b3(y3x, training=training)
        y3x = self.f1p3(y3x)
        y = self.f1concat([y1x, y3x])

        # Fire 2
        y = self.f2c1(y)
        y = self.f2b1(y, training=training)
        y = self.f2p1(y)
        y1x = self.f2c2(y)
        y1x = self.f2b2(y1x, training=training)
        y1x = self.f2p2(y1x)
        y3x = self.f2c3(y)
        y3x = self.f2b3(y3x, training=training)
        y3x = self.f2p3(y3x)
        y = self.f2concat([y1x, y3x])

        # Fire 3
        y = self.f3c1(y)
        y = self.f3b1(y, training=training)
        y = self.f3p1(y)
        y1x = self.f3c2(y)
        y1x = self.f3b2(y1x, training=training)
        y1x = self.f3p3(y1x)
        y3x = self.f3c3(y)
        y3x = self.f3b3(y3x, training=training)
        y3x = self.f3p3(y3x)
        y = self.f3concat([y1x, y3x])

        y = self.mp3(y)

        # Fire 4
        y = self.f4c1(y)
        y = self.f4b1(y, training=training)
        y = self.f4p1(y)
        y1x = self.f4c2(y)
        y1x = self.f4b2(y1x, training=training)
        y1x = self.f4p2(y1x)
        y3x = self.f4c3(y)
        y3x = self.f4b3(y3x, training=training)
        y3x = self.f4p3(y3x)
        y = self.f4concat([y1x, y3x])

        # Fire 5
        y = self.f5c1(y)
        y = self.f5b1(y, training=training)
        y = self.f5p1(y)
        y1x = self.f5c2(y)
        y1x = self.f5b2(y1x, training=training)
        y1x = self.f5p2(y1x)
        y3x = self.f5c3(y)
        y3x = self.f5b3(y3x, training=training)
        y3x = self.f5p3(y3x)
        y = self.f5concat([y1x, y3x])

        # Fire 6
        y = self.f6c1(y)
        y = self.f6b1(y, training=training)
        y = self.f6p1(y)
        y1x = self.f6c2(y)
        y1x = self.f6b2(y1x, training=training)
        y1x = self.f6p2(y1x)
        y3x = self.f6c3(y)
        y3x = self.f6b3(y3x, training=training)
        y3x = self.f6p3(y3x)
        y = self.f6concat([y1x, y3x])

        # Fire 7
        y = self.f7c1(y)
        y = self.f7b1(y, training=training)
        y = self.f7p1(y)
        y1x = self.f7c2(y)
        y1x = self.f7b2(y1x, training=training)
        y1x = self.f7p2(y1x)
        y3x = self.f7c3(y)
        y3x = self.f7b3(y3x, training=training)
        y3x = self.f7p3(y3x)
        y = self.f7concat([y1x, y3x])

        y = self.mp7(y)

        # Fire 8
        y = self.f8c1(y)
        y = self.f8b1(y, training=training)
        y = self.f8p1(y)
        y1x = self.f8c2(y)
        y1x = self.f8b2(y1x, training=training)
        y1x = self.f8p2(y1x)
        y3x = self.f8c3(y)
        y3x = self.f8b3(y3x, training=training)
        y3x = self.f8p3(y3x)
        y = self.f8concat([y1x, y3x])

        y = self.avgpool(y)
        y = self.classifier(y)
        tf.compat.v1.summary.histogram('output', y)

        return y
