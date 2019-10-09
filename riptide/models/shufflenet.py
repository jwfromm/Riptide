import os
import tensorflow as tf
from riptide.binary import binary_layers as nn

def channel_shuffle(x, groups):
    n, h, w, c = x.shape
    channels_per_group = tf.math.floordiv(c, groups)

    # reshape
    x = tf.reshape(x, [n, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [n, h, w, c])

    return x

