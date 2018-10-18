import tensorflow as tf
import tensorflow.keras as keras

def get_quantize_bits(x):
    if len(x.shape) > 1:
        mean = tf.reduce_mean(tf.abs(tf.reshape(x, [x.shape[0], -1])), axis=-1)
    else:
        mean = tf.reduce_mean(tf.abs(x))
    # Fix dimensions of mean
    for i in range(len(x.shape) - 1):
        mean = tf.expand_dims(mean, axis=-1)
    bits = tf.cast(x >= 0, tf.float32)
    bits = (2*bits) - 1
    return mean, bits

@tf.custom_gradient
def Quantize(x):
    mean, bits = get_quantize_bits(x)
    y = mean * bits

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
