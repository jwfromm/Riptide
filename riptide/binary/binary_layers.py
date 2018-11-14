import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.framework import common_shapes
from riptide.binary.binary_funcs import *
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Activation, PReLU, BatchNormalization, MaxPool2D
"""Quantization scope, defines the modification of operator"""


class Config(object):
    """Configuration scope of current mode.

    This is used to easily switch between different
    model structure variants by simply calling into these functions.

    Parameters
    ----------
    actQ : function
        Activation quantization

    weightQ : function: name->function
        Maps name to quantize function.

    bits : Tensor
        When using HWGQ binarization, these are the possible values
        that can be used in approximation. For other binarization schemes,
        this should be the number of bits to use.

    Example
    -------
    import qnn

    with qnn.Config(actQ=qnn.quantize(bits=8, scale=8, signed=True),
                    weightQ=qnn.identity,
                    use_bn=True):
        net = qnn.get_model(model_name, **kwargs)
    """
    current = None

    def __init__(self, actQ=None, weightQ=None, activation=None, bits=None, use_bn=False):
        self.actQ = actQ if actQ else lambda x: x
        self.weightQ = weightQ if weightQ else lambda x: x
        self.bits = bits
        self.use_bn = use_bn

    def __enter__(self):
        self._old_manager = Config.current
        Config.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Config.current = self._old_manager


class Conv2D(keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.actQ = self.scope.actQ
        self.weightQ = self.scope.weightQ
        self.bits = self.scope.bits
        self.use_bn = self.scope.use_bn
        if self.use_bn:
            self.bn = BatchNormalization()

    def call(self, inputs):
        with tf.name_scope("actQ"):
            if self.bits is not None:
                inputs = self.actQ(inputs, self.bits)
            else:
                inputs = self.actQ(inputs)
            tf.summary.histogram(tf.contrib.framework.get_name_scope(), inputs)
        with tf.name_scope("weightQ"):
            kernel = self.weightQ(self.kernel)
            tf.summary.histogram(tf.contrib.framework.get_name_scope(), kernel)
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        
        if self.use_bn:
            outputs = self.bn(outputs)
            
        return outputs


# Same as default keras conv2d but has batchnorm build in with Config.
class Conv2DBatchNorm(keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.use_bn = self.scope.use_bn
        if self.use_bn:
            self.bn = BatchNormalization()

    def call(self, inputs):
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        
        if self.use_bn:
            outputs = self.bn(outputs)
            
        return outputs
    

class Dense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.actQ = self.scope.actQ
        self.weightQ = self.scope.weightQ
        self.bits = self.scope.bits
        self.use_bn = self.scope.use_bn
        if self.use_bn:
            self.bn = BatchNormalization()

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        with tf.name_scope("actQ"):
            if self.bits is not None:
                inputs = self.actQ(inputs, self.bits)
            else:
                inputs = self.actQ(inputs)
            tf.summary.histogram(tf.contrib.framework.get_name_scope(), inputs)
        with tf.name_scope("weightQ"):
            kernel = self.weightQ(self.kernel)
            tf.summary.histogram(tf.contrib.framework.get_name_scope(), kernel)
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
            return self.activation(outputs)  # pylint: disable=not-callable
        if self.use_bn:
            outputs = self.bn(outputs)
        return outputs


class Scalu(keras.layers.Layer):
    def __init__(self):
        super(Scalu, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight(
            'scale', shape=[1], initializer='ones', trainable=True)

    def call(self, input):
        return input * self.scale


NormalConv2D = keras.layers.Conv2D
NormalDense = keras.layers.Dense
