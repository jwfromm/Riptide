import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from .bit_approximations import load_clusters, load_bits


def log2(x):
    return tf.log(x) / tf.log(2.0)


@tf.custom_gradient
def AP2(x):
    #x = tf.clip_by_value(x, 1e-7, 1.0)
    # Positive ap2 might be fine
    with tf.name_scope("AP2"):
        y = 2**(tf.round(log2(tf.abs(x))))

    def grad_fn(dy):
        return [dy]

    return y, grad_fn


def get_numpy(sess, x):
    if not isinstance(x, list):
        x = [x]
    with sess.as_default():
        output = sess.run(x)
    if len(output) == 1:
        output = output[0]
    return output


def get_shiftnorm_ap2(sess, layer, pure_shiftnorm=True, conv_weights=None):
    mean = layer.weights[0].value()
    extra_scale = layer.extra_scale
    epsilon = layer.epsilon
    if pure_shiftnorm:
        approximate_mean = AP2((extra_scale * mean + epsilon))
        with sess.as_default():
            return approximate_mean.eval(), None

    else:
        variance = layer.weights[1].value()
        mean_scale = 1.0 + (1 / (2**layer.bits - 1))
        approximate_std = AP2(1.0 /
                              (extra_scale * tf.sqrt(variance + epsilon)))
        if conv_weights is not None:
            weight_scale_ap2, _ = get_quantize_bits(conv_weights)
        else:
            weight_scale_ap2 = 1.0
        weight_scale_bits = log2(1.0 / weight_scale_ap2)
        shiftnorm_scale_bits = log2(1.0 / approximate_std)
        total_shift_bits = weight_scale_bits + shifnorm_scale_bits + layer.bits
        quantized_mean = FixedPointQuantize(mean, mean_scale, total_shift_bits)
        with sess.as_default():
            return approximate_std.eval(), quantized_mean.eval()


def get_quantize_bits(x):
    if len(x.shape) > 1:
        mean = tf.reduce_mean(tf.abs(tf.reshape(x, [-1, x.shape[-1]])), axis=0)
    else:
        mean = tf.reduce_mean(tf.abs(x))
    # Fix dimensions of mean
    for i in range(len(x.shape) - 1):
        mean = tf.expand_dims(mean, axis=0)
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2 * bits) - 1
    return AP2(mean), bits


# Fixed point quantize operator that supports per_channel scales and bitwidth.
# Assumes min and max value are both scale.
@tf.custom_gradient
def FixedPointQuantize(inputs, scale, bits, rescale=True):
    # Start by clipping values between specified range.
    y = tf.clip_by_value(inputs, -scale, scale)
    # Determine floating point value of each bit.
    bit_value = scale / (2.0**bits - 1.0)
    # Quantize tensor.
    y = y / bit_value
    y = tf.round(y)
    # Readjust to floating point
    if rescale:
        y = y * bit_value

    def grad_fn(dy):
        grad_mask = tf.cast(tf.abs(inputs) <= scale, tf.float32)
        dx = grad_mask * dy
        return [dx]

    return y, grad_fn


@tf.custom_gradient
def XQuantize(x):
    mean, bits = get_quantize_bits(x)
    y = mean * bits

    def grad_fn(dy):
        #grad_mask_greater = tf.cast(tf.abs(x) >= 1, tf.float32)
        #grad_mask_lesser = tf.cast(tf.abs(x) <= 1, tf.float32)
        # Let big values leak a little
        #grad_mask = 0.1 * grad_mask_greater + grad_mask_lesser
        grad_mask = tf.cast(tf.abs(x) <= 1, tf.float32)
        dx = grad_mask * dy
        return [dx]

    return y, grad_fn


@tf.custom_gradient
def Quantize(x):
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2 * bits) - 1

    y = bits

    def grad_fn(dy):
        #grad_mask_greater = tf.cast(tf.abs(x) >= 1, tf.float32)
        #grad_mask_lesser = tf.cast(tf.abs(x) <= 1, tf.float32)
        # Let big values leak a little
        #grad_mask = 0.1 * grad_mask_greater + grad_mask_lesser
        grad_mask = tf.cast(tf.abs(x) <= 1, tf.float32)
        dx = grad_mask * dy
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
