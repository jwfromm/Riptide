import tensorflow as tf
import tensorflow.keras as keras
from .bit_approximations import load_clusters, load_bits


def log2(x):
    return tf.log(x)/tf.log(2.0)


def AP2(x):
    x = tf.clip_by_value(x, 1e-7, 1)
    return 2**(tf.round(log2(tf.abs(x))))


def get_quantize_bits(x):
    if len(x.shape) > 1:
        mean = tf.reduce_mean(tf.abs(tf.reshape(x, [x.shape[0], -1])), axis=-1)
    else:
        mean = tf.reduce_mean(tf.abs(x))
    # Fix dimensions of mean
    for i in range(len(x.shape) - 1):
        mean = tf.expand_dims(mean, axis=-1)
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2 * bits) - 1
    return AP2(mean), bits


@tf.custom_gradient
def XQuantize(x):
    mean, bits = get_quantize_bits(x)
    y = mean * bits

    def grad_fn(dy):
        dx = dy * tf.cast(tf.abs(x) <= 1, tf.float32)
        return [dx]

    return y, grad_fn


@tf.custom_gradient
def Quantize(x):
    y = tf.sign(x)

    def grad_fn(dy):
        dx = dy * tf.cast(tf.abs(x) <= 1, tf.float32)
        return [dx]

    return y, grad_fn


def get_HWGQ_bits(x, clusters):
    # Computes HWG quantization and returns the integer binary value.
    for i in range(len(x.shape)):
        # need to reshape clusters properly.
        clusters = tf.expand_dims(clusters, axis=0)
    # Add new data axis for proper subtraction.
    x = tf.expand_dims(x, axis=-1)

    # Compute best fitting cluster for each value in data.
    distance = tf.abs(x - clusters)
    indices = tf.argmin(distance, axis=-1)
    return indices


@tf.custom_gradient
def HWGQuantize(x, clusters):
    indices = get_HWGQ_bits(x, clusters)
    y = tf.gather(clusters, indices)

    def grad_fn(dy):
        max_cluster = tf.reduce_max(clusters)
        min_cluster = tf.reduce_min(clusters)
        grad_filter = tf.logical_and(min_cluster <= x, x <= max_cluster)
        dx = dy * tf.cast(grad_filter, tf.float32)
        return [dx, None]

    return y, grad_fn


# Assumes input is clipped to [0, 1]
@tf.custom_gradient
def DQ(x, bits):
    output = (1.0 / (2.0**bits - 1.0)) * tf.round((2.0**bits - 1.0) * x)

    def grad_fn(dy):
        return [dy, None]

    return output, grad_fn


def DQuantize(x, bits):
    x = tf.clip_by_value(x, 0, 1)
    return DQ(x, bits)


def DQuantizeW(x, bits):
    x = tf.tanh(x) / (2.0 * tf.reduce_max(tf.abs(tf.tanh(x)))) + 0.5
    return (2. * DQuantize(x, bits)) - 1.0


def DQuantizeBits(x, bits):
    x = tf.clip_by_value(x, 0, 1)
    return tf.round(x * (2.0**bits - 1.0))


def DQuantizeBitsW(x, bits):
    shifted_x = (tf.tanh(x) / (2.0 * tf.reduce_max(tf.abs(tf.tanh(x))))) + 0.5
    return DQuantizeBits(shifted_x)


# Takes bit value x and converts it to floating point approximation.
def DBits2Value(x, bits):
    return x / (2.0**bits - 1.0)


def DBits2ValueW(x, bits):
    approx = DBits2Value(x, bits)
    return 2.0 * (approx - 0.5)
