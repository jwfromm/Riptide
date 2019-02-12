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


def compute_quantized_shiftnorm(variance,
                                mean,
                                epsilon,
                                conv_weights,
                                extra_scale,
                                bits,
                                rescale=True):
    # Compute number of bits to shift.
    std_factor = (1.0 / (extra_scale * tf.sqrt(variance + epsilon)))
    with tf.name_scope('AP2'):
        approximate_std = AP2(std_factor)
    # Now determine number of bits needed, the sum of weight scale
    # bits and shift norm scale bits.
    weight_scale_ap2, _ = get_quantize_bits(conv_weights)
    weight_scale_bits = -log2(weight_scale_ap2)
    weight_scale_bits = tf.reshape(weight_scale_bits, [-1])
    total_shift_bits = weight_scale_bits + bits

    # Quantizing the mean is a little tricky, start by determining
    # the quantization scale.
    mean_scale = 1.0 + ((1.0 / (2.0**bits - 1.0)) *
                        (1.0 - (1.0 / 2.0**weight_scale_bits)))

    # Now quantize each channel of mean appropriately.
    with tf.name_scope('FPQ'):
        quantized_means = FixedPointQuantize(mean, mean_scale,
                                             total_shift_bits, rescale)
    return approximate_std, quantized_means


def get_shiftnorm_ap2(layer, conv_weights, rescale=False):
    mean = layer.weights[0].value()
    extra_scale = layer.extra_scale
    epsilon = layer.epsilon
    variance = layer.weights[1].value()
    bits = layer.bits
    approximate_std, quantized_means = compute_quantized_shiftnorm(
        variance, mean, epsilon, conv_weights, extra_scale, bits, rescale)
    return approximate_std, quantized_means


def get_quantize_bits(x):
    if len(x.shape) > 2:
        mean = tf.reduce_mean(tf.abs(tf.reshape(x, [-1, x.shape[-1]])), axis=0)
    else:
        mean = tf.reduce_mean(tf.abs(x))
    # Fix dimensions of mean
    for i in range(len(x.shape) - 1):
        mean = tf.expand_dims(mean, axis=0)
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2 * bits) - 1
    with tf.name_scope("AP2"):
        approximate_mean = AP2(mean)
    return approximate_mean, bits


# Fixed point quantize operator that supports per_channel scales and bitwidth.
# Assumes min and max value are both scale.
@tf.custom_gradient
def FixedPointQuantize(inputs, scale, bits, rescale):
    # Start by clipping values between specified range.
    y = tf.clip_by_value(inputs, -scale, scale)
    # Determine floating point value of each bit.
    bit_value = scale / (2.0**bits - 1.0)
    # Quantize tensor.
    y = y / bit_value
    y = tf.round(y)
    # Readjust to floating point if specified.
    y = tf.cond(rescale, true_fn=lambda: y * bit_value, false_fn=lambda: y)

    def grad_fn(dy):
        grad_mask = tf.cast(tf.abs(inputs) <= scale, tf.float32)
        dx = grad_mask * dy
        return [dx, None, None, None]

    return y, grad_fn


@tf.custom_gradient
def XQuantize(x):
    mean, bits = get_quantize_bits(x)
    y = mean * bits

    def grad_fn(dy):
        # Use a larger gradient cutoff to allow weights to grow if needed.
        # This can effect the scales based on the means of kernels.
        # Likely has no significant effect though.
        gradient_cutoff = 10.0
        grad_mask = tf.cast(tf.abs(x) <= gradient_cutoff, tf.float32)
        # Allow weights to move off away from 1 if needed.
        leaky_grad_mask = tf.cast(
            tf.logical_or(
                tf.logical_and(x > gradient_cutoff, dy > 0),
                tf.logical_and(x < -gradient_cutoff, dy < 0)), tf.float32)
        dx = grad_mask * dy + 0.1 * leaky_grad_mask * dy
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
def DQ(x, bits, bipolar):
    # Use small adjustment to avoid rounding inconsistency.
    if bipolar:
        # Convert incoming [-1, 1] range to [0, 1]
        x = (x + 1.0) / 2.0

    epsilon = 1e-5
    # Round to nearest linear bin in [0, 1].
    output = (1.0 /
              (2.0**bits - 1.0)) * tf.round((2.0**bits - 1.0) * x + epsilon)

    if bipolar:
        # Deconvert back to [-1, 1]
        output = (output - 0.5) * 2.0

    # Pass through gradient.
    def grad_fn(dy):
        return [dy, None]

    return output, grad_fn


def DQuantize(x, bits, bipolar=False):
    # Apply clipping in [0, 1] with associated gradient.
    if bipolar:
        x = tf.clip_by_value(x, -1, 1)
    else:
        x = tf.clip_by_value(x, 0, 1)

    # Quantize linearlly.
    return DQ(x, bits, bipolar)


def DQuantizeW(x, bits):
    x = tf.tanh(x) / (2.0 * tf.reduce_max(tf.abs(tf.tanh(x)))) + 0.5
    return (2. * DQuantize(x, bits)) - 1.0


def DQuantizeBits(x, bits, bipolar=False):
    if bipolar:
        x = tf.clip_by_value(x, -1, 1)
    else:
        x = tf.clip_by_value(x, 0, 1)
    epsilon = 1e-5
    if bipolar:
        # Convert from [-1, 1] to [0, 1] for rounding
        x = (x + 1.0) / 2.0
    return tf.round(x * (2.0**bits - 1.0) + epsilon)


def DQuantizeBitsW(x, bits):
    shifted_x = (tf.tanh(x) / (2.0 * tf.reduce_max(tf.abs(tf.tanh(x))))) + 0.5
    return DQuantizeBits(shifted_x)


# Takes bit value x and converts it to floating point approximation.
def DBits2Value(x, bits):
    return x / (2.0**bits - 1.0)


def DBits2ValueW(x, bits):
    approx = DBits2Value(x, bits)
    return 2.0 * (approx - 0.5)
