import tensorflow as tf
import tensorflow.keras.layers as nn

def FireLayer(inputs, squeeze_channels, expand1x1_channels, expand3x3_channels):
    f = _make_fire_conv(inputs, squeeze_channels, 1)
    f1x1 = _make_fire_conv(f, expand1x1_channels, 1)
    f3x3 = _make_fire_conv(f, expand3x3_channels, 3, 'same')
    fout = nn.Concatenate(axis=-1)([f1x1, f3x3])
    return fout

def _make_fire_conv(inputs, channels, kernel_size, padding='valid'):
    out = nn.Conv2D(channels, kernel_size, padding=padding)(inputs)
    out = nn.Activation('relu')(out)
    return out

def SqueezeNet(classes=1000):
    inputs = nn.Input(shape=[224, 224, 3])
    x = nn.Conv2D(64, kernel_size=3, strides=2)(inputs)
    x = nn.Activation('relu')(x)
    x = nn.MaxPool2D(pool_size=3, strides=2)(x)
    x = FireLayer(x, 16, 64, 64)
    x = FireLayer(x, 16, 64, 64)
    x = nn.MaxPool2D(pool_size=3, strides=2)(x)
    x = FireLayer(x, 32, 128, 128)
    x = FireLayer(x, 32, 128, 128)
    x = nn.MaxPool2D(pool_size=3, strides=2)(x)
    x = FireLayer(x, 48, 192, 192)
    x = FireLayer(x, 48, 192, 192)
    x = FireLayer(x, 64, 256, 256)
    x = FireLayer(x, 64, 256, 256)
    x = FireLayer(x, 64, 256, 256)
    x = nn.Dropout(0.5)(x)

    x = nn.Conv2D(classes, kernel_size=1)(x)
    x = nn.Activation('relu')(x)
    x = nn.AvgPool2D(13)(x)
    x = nn.Flatten()(x)

    return tf.keras.models.Model(inputs=inputs, outputs=x)
