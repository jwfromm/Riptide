import tensorflow as tf
from riptide.binary import binary_layers as nn
#import tensorflow.keras.layers as nn

bnmomemtum=0.9

class SqueezeNet(tf.keras.Model):
    def __init__(self, classes=1000):
        super(SqueezeNet, self).__init__()
        self.classes = classes

        self.c1 = nn.NormalConv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')
        self.b1 = nn.NormalBatchNormalization(momentum=bnmomemtum)

        self.f1c1 = nn.BinaryConv2D(filters=24, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f1b1 = nn.BatchNormalization(self.f1c1, momentum=bnmomemtum)
        self.f1c2 = nn.BinaryConv2D(filters=48//2, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f1b2 = nn.BatchNormalization(self.f1c2, momentum=bnmomemtum)
        self.f1c3 = nn.BinaryConv2D(filters=48//2, kernel_size=3, activation='relu', padding='same', use_bias=False)
        self.f1b3 = nn.BatchNormalization(self.f1c3, momentum=bnmomemtum)
        self.f1concat = nn.Concatenate(axis=-1)

        self.mp1 = nn.MaxPool2D(pool_size=2)

        self.f2c1 = nn.BinaryConv2D(filters=48, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f2b1 = nn.BatchNormalization(self.f2c1, momentum=bnmomemtum)
        self.f2c2 = nn.BinaryConv2D(filters=96//2, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f2b2 = nn.BatchNormalization(self.f2c2, momentum=bnmomemtum)
        self.f2c3 = nn.BinaryConv2D(filters=96//2, kernel_size=3, activation='relu', padding='same', use_bias=False)
        self.f2b3 = nn.BatchNormalization(self.f2c3, momentum=bnmomemtum)
        self.f2concat = nn.Concatenate(axis=-1)

        self.mp2 = nn.MaxPool2D(pool_size=2)

        self.f3c1 = nn.BinaryConv2D(filters=64, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f3b1 = nn.BatchNormalization(self.f3c1, momentum=bnmomemtum)
        self.f3c2 = nn.BinaryConv2D(filters=128//2, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f3b2 = nn.BatchNormalization(self.f3c2, momentum=bnmomemtum)
        self.f3c3 = nn.BinaryConv2D(filters=128//2, kernel_size=3, activation='relu', padding='same', use_bias=False)
        self.f3b3 = nn.BatchNormalization(self.f3c3, momentum=bnmomemtum)
        self.f3concat = nn.Concatenate(axis=-1)

        self.mp3 = nn.MaxPool2D(pool_size=2)

        self.f4c1 = nn.BinaryConv2D(filters=48, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f4b1 = nn.BatchNormalization(self.f4c1, momentum=bnmomemtum)
        self.f4c2 = nn.BinaryConv2D(filters=96//2, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f4b2 = nn.BatchNormalization(self.f4c2, momentum=bnmomemtum)
        self.f4c3 = nn.BinaryConv2D(filters=96//2, kernel_size=3, activation='relu', padding='same', use_bias=False)
        self.f4b3 = nn.BatchNormalization(self.f4c3, momentum=bnmomemtum)
        self.f4concat = nn.Concatenate(axis=-1)

        self.mp4 = nn.MaxPool2D(pool_size=2)

        self.f5c1 = nn.BinaryConv2D(filters=24, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f5b1 = nn.BatchNormalization(self.f5c1, momentum=bnmomemtum)
        self.f5c2 = nn.BinaryConv2D(filters=48//2, kernel_size=1, activation='relu', padding='same', use_bias=False)
        self.f5b2 = nn.BatchNormalization(self.f5c2, momentum=bnmomemtum)
        self.f5c3 = nn.BinaryConv2D(filters=48//2, kernel_size=3, activation='relu', padding='same', use_bias=False)
        self.f5b3 = nn.BatchNormalization(self.f5c3, momentum=bnmomemtum)
        self.f5concat = nn.Concatenate(axis=-1)

        self.avgpool = nn.GlobalAveragePooling2D()
        self.classifier = nn.NormalDense(1000, use_bias=False)
        #self.scalu = nn.Scalu(scale=0.1)
        self.softmax = nn.Activation('softmax')


    def call(self, x, training=None):
        y = self.c1(x)
        y = self.b1(y, training=training)

        y = self.f1c1(y)
        y = self.f1b1(y, training=training)
        y1x = self.f1c2(y)
        y1x = self.f1b2(y1x, training=training)
        y3x = self.f1c3(y)
        y3x = self.f1b3(y3x, training=training)
        y = self.f1concat([y1x, y3x])

        y = self.mp1(y)

        y = self.f2c1(y)
        y = self.f2b1(y, training=training)
        y1x = self.f2c2(y)
        y1x = self.f2b2(y1x, training=training)
        y3x = self.f2c3(y)
        y3x = self.f2b3(y3x, training=training)
        y = self.f2concat([y1x, y3x])

        y = self.mp2(y)

        y = self.f3c1(y)
        y = self.f3b1(y, training=training)
        y1x = self.f3c2(y)
        y1x = self.f3b2(y1x, training=training)
        y3x = self.f3c3(y)
        y3x = self.f3b3(y3x, training=training)
        y = self.f3concat([y1x, y3x])

        y = self.mp3(y)

        y = self.f4c1(y)
        y = self.f4b1(y, training=training)
        y1x = self.f4c2(y)
        y1x = self.f4b2(y1x, training=training)
        y3x = self.f4c3(y)
        y3x = self.f4b3(y3x, training=training)
        y = self.f4concat([y1x, y3x])

        y = self.mp4(y)

        y = self.f5c1(y)
        y = self.f5b1(y, training=training)
        y1x = self.f5c2(y)
        y1x = self.f5b2(y1x, training=training)
        y3x = self.f5c3(y)
        y3x = self.f5b3(y3x, training=training)
        y = self.f5concat([y1x, y3x])

        y = self.avgpool(y)
        y = self.classifier(y)
        #y = self.scalu(y)
        y = self.softmax(y)
        tf.compat.v1.summary.histogram('output', y)

        return y
