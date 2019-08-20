import tensorflow as tf
import tensorflow.keras.layers as nn

class FireLayer(tf.keras.layers.Layer):
    def __init__(self, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireLayer, self).__init__()
        self.fire = _make_fire_conv(squeeze_channels, 1)
        self.fire1x1 = _make_fire_conv(expand1x1_channels, 1)
        self.fire3x3 = _make_fire_conv(expand3x3_channels, 3, 'same')

    def call(self, x):
        x = self.fire(x)
        x1 = self.fire1x1(x)
        x3 = self.fire3x3(x)

        x = tf.concat([x1, x3], axis=-1)
        return x

def _make_fire_conv(channels, kernel_size, padding='valid'):
    out = tf.keras.models.Sequential()
    out.add(nn.Conv2D(channels, kernel_size, padding=padding))
    out.add(nn.Activation('relu'))
    return out

class SqueezeNet(tf.keras.Model):
    def __init__(self, classes=1000):
        super(SqueezeNet, self).__init__()
        self.features = tf.keras.models.Sequential()
        self.features.add(nn.Conv2D(64, kernel_size=3, strides=2))
        self.features.add(nn.Activation('relu'))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
        self.features.add(FireLayer(16, 64, 64))
        self.features.add(FireLayer(16, 64, 64))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
        self.features.add(FireLayer(32, 128, 128))
        self.features.add(FireLayer(32, 128, 128))
        self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
        self.features.add(FireLayer(48, 192, 192))
        self.features.add(FireLayer(48, 192, 192))
        self.features.add(FireLayer(64, 256, 256))
        self.features.add(FireLayer(64, 256, 256))
        self.features.add(FireLayer(64, 256, 256))
        self.features.add(nn.Dropout(0.5))

        self.classifier = tf.keras.models.Sequential()
        self.classifier.add(nn.Conv2D(classes, kernel_size=1))
        self.classifier.add(nn.Activation('relu'))
        self.classifier.add(nn.AvgPool2D(13))
        self.classifier.add(nn.Flatten())

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
