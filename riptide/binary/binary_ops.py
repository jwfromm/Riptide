import tensorflow as tf
import numpy as np
from tensorflow.python.ops import bitwise_ops

def bitwise_xnor(a, b):
    # Need to do some dim expanding to handle batches.
    a = tf.expand_dims(a, axis=1)
    b = tf.expand_dims(b, axis=0)
    ab = bitwise_ops.invert(bitwise_ops.bitwise_xor(a, b))
    return ab

def binarize_dense(x, transpose=False):
    if transpose:
        x = tf.transpose(x, [1,0])
    h, w = x.shape
    num_bins = int(w / 64)
    binary_x = tf.cast(x > 0, tf.int64)
    packed_x= []
    for b in range(num_bins):
        packed_x.append(tf.zeros(shape=[h], dtype=tf.int64))
    for k in range(num_bins):
        for b in range(64):
            packed_x[k] = bitwise_ops.bitwise_or(packed_x[k], bitwise_ops.left_shift(binary_x[:, 64*k + b], b))
    packed_x = tf.stack(packed_x, axis=-1)     
    return packed_x

def binarize_dense_fast(x, transpose=False):
    if transpose:
        x = tf.transpose(x, [1,0])
    h, w = x.shape
    num_bins = int(w / 64)
    # Create shift tensor and apply it to binarized input.
    shift_bits = tf.broadcast_to(tf.range(64, dtype=tf.int64), x.shape)
    binary_x = tf.cast(x > 0, tf.int64)
    binary_x = bitwise_ops.left_shift(binary_x, shift_bits)
    # Split binarized x into chunks.
    binary_chunks = tf.split(binary_x, num_bins, axis=-1)
    # Combine chunks using bitwise or (equivalent to reduce sum).
    packed_x = tf.reduce_sum(binary_chunks, axis=-1)
    packed_x = tf.transpose(packed_x, [1,0])
    return packed_x
    
def binary_dense_matmul(a, b):
    ab = bitwise_xnor(a, b)
    pcnt = 2*(tf.cast(bitwise_ops.population_count(ab), tf.float32)) - 64
    inner_sum = tf.reduce_sum(pcnt, axis=-1)
    return inner_sum

def binary_dense(a, b, binarize_a=True, binarize_b=False):
    if binarize_a:
        bin_a = binarize_dense_fast(a)
    else:
        bin_a = a
    if binarize_b:
        bin_b = binarize_dense_fast(b, transpose=True)
    else:
        bin_b = tf.transpose(b, [1,0])
    return binary_dense_matmul(bin_a, bin_b)