import tensorflow as tf
from riptide.anneal.anneal_config import Config
from tensorflow.python.framework import common_shapes


@tf.custom_gradient
def AlphaClip(x, alpha):
    output = tf.clip_by_value(x, 0, alpha)

    def grad_fn(dy):
        x_grad_mask = tf.cast(tf.logical_and(x >= 0, x <= alpha), tf.float32)
        alpha_grad_mask = tf.cast(x >= alpha, tf.float32)
        alpha_grad = tf.reduce_sum(dy * alpha_grad_mask)
        x_grad = dy * x_grad_mask

        return [x_grad, alpha_grad]

    return output, grad_fn


@tf.custom_gradient
def AlphaQuantize(x, alpha, bits):
    output = tf.round(x * ((2**bits - 1) / alpha)) * (alpha / (2**bits - 1))

    def grad_fn(dy):
        return [dy, None, None]

    return output, grad_fn


class PACT(tf.keras.layers.Layer):
    def __init__(self):
        super(PACT, self).__init__()
        self.scope = Config.current
        self.quantize = self.scope.quantize
        self.bits = self.scope.a_bits

    def build(self, input_shape):
        if self.quantize:
            self.alpha = self.add_variable(
                'alpha',
                shape=[],
                initializer=tf.initializers.Constant([10.]),
                regularizer=tf.keras.regularizers.l2(0.0002))

    def call(self, inputs):
        if self.quantize:
            outputs = AlphaClip(inputs, self.alpha)
            tf.compat.v1.summary.histogram('alpha', self.alpha)
            with tf.name_scope('QA'):
                outputs = AlphaQuantize(outputs, self.alpha, self.bits)
                tf.compat.v1.summary.histogram('activation', inputs)
                tf.compat.v1.summary.histogram('quantized_activation', outputs)
        else:
            outputs = tf.nn.relu(inputs)
        return outputs

    def get_config(self):
        return {'quantize': self.quantize, 'bits': self.bits}

    def compute_output_shape(self, input_shape):
        return input_shape


def get_sawb_coefficients(bits):
    bits = int(bits)
    assert bits <= 4, "Currently only supports bitwidths up to 4."
    coefficient_dict = {
        1: [0., 1.],
        2: [3.19, -2.14],
        3: [7.40, -6.66],
        4: [11.86, -11.68]
    }
    return coefficient_dict[bits]


@tf.custom_gradient
def SAWBQuantize(x, alpha, bits):
    # Clip between -alpha and alpha
    clipped = tf.clip_by_value(x, -alpha, alpha)
    # Rescale to [0, alpha]
    scaled = (clipped + alpha) / 2.
    # Quantize.
    quantized = tf.round(scaled * ((2**bits - 1) / alpha)) * (alpha /
                                                              (2**bits - 1))
    # Rescale to negative range.
    output = (2 * quantized) - alpha

    def grad_fn(dy):
        return [dy, None, None]

    return output, grad_fn


class SAWBConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(SAWBConv2D, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.quantize = self.scope.quantize
        self.bits = self.scope.w_bits
        if self.quantize:
            self.c1, self.c2 = get_sawb_coefficients(self.bits)

    def call(self, inputs):
        if self.quantize:
            # Compute proper scale for our weights.
            alpha = self.c1 * tf.sqrt(tf.reduce_mean(
                self.kernel**2)) + self.c2 * tf.reduce_mean(
                    tf.abs(self.kernel))
            # Quantize kernel
            with tf.name_scope("QW"):
                kernel = SAWBQuantize(self.kernel, alpha, self.bits)
                tf.compat.v1.summary.histogram("weight", self.kernel)
                tf.compat.v1.summary.histogram("quantized_weight", kernel)
        else:
            kernel = self.kernel

        # Invoke convolution
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class SAWBDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(SAWBDense, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.quantize = self.scope.quantize
        self.bits = self.scope.w_bits
        if self.quantize:
            self.c1, self.c2 = get_sawb_coefficients(self.bits)

    def call(self, inputs):
        if self.quantize:
            alpha = self.c1 * tf.sqrt(tf.reduce_mean(
                self.kernel**2)) + self.c2 * tf.reduce_mean(
                    tf.abs(self.kernel))
            with tf.name_scope("QW"):
                kernel = SAWBQuantize(self.kernel, alpha, self.bits)
                tf.compat.v1.summary.histogram("weight", self.kernel)
                tf.compat.v1.summary.histogram("quantized_weight", kernel)
        else:
            kernel = self.kernel

        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
