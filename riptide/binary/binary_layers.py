import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.framework import common_shapes
from .binary_funcs import *
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Activation, PReLU
from .normalization import ShiftNormalization
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

    use_bn: bool
        Whether to apply batch normalization at the end of the layer or not.

    pure_shiftnorm: bool
        If true, shift normalization only scales and has no centering. Truly
        only shifting.

    use_qadd: bool
        If true, do quantization before addition in Qadd layers.

    Example
    -------
    import qnn

    with qnn.Config(actQ=qnn.quantize(bits=8, scale=8, signed=True),
                    weightQ=qnn.identity,
                    use_bn=True):
        net = qnn.get_model(model_name, **kwargs)
    """
    current = None

    def __init__(self,
                 actQ=None,
                 weightQ=None,
                 bits=None,
                 use_bn=True,
                 use_maxpool=True,
                 use_act=True,
                 pure_shiftnorm=True,
                 use_qadd=False):
        self.actQ = actQ if actQ else lambda x: x
        self.weightQ = weightQ if weightQ else lambda x: x
        self.bits = bits
        self.use_bn = use_bn
        self.use_act = use_act
        self.pure_shiftnorm = pure_shiftnorm
        self.use_qadd = use_qadd
        self.use_maxpool = use_maxpool

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
        self.use_act = self.scope.use_act

    def call(self, inputs):
        with tf.name_scope("actQ"):
            tf.summary.histogram('prebinary_activations', inputs)
            if self.bits is not None:
                inputs = self.actQ(inputs, self.bits)
            else:
                inputs = self.actQ(inputs)
            tf.summary.histogram('binary_activations', inputs)
        with tf.name_scope("weightQ"):
            kernel = self.weightQ(self.kernel)
            tf.summary.histogram('weights', self.kernel)
            tf.summary.histogram('binary_weights', kernel)
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format='NHWC')

        if self.use_act and self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


class Dense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.scope = Config.current
        self.actQ = self.scope.actQ
        self.weightQ = self.scope.weightQ
        self.bits = self.scope.bits
        self.use_act = self.scope.use_act

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        with tf.name_scope("actQ"):
            tf.summary.histogram('prebinary_activations', inputs)
            if self.bits is not None:
                inputs = self.actQ(inputs, self.bits)
            else:
                inputs = self.actQ(inputs)
            tf.summary.histogram('binary_activations', inputs)
        with tf.name_scope("weightQ"):
            kernel = self.weightQ(self.kernel)
            tf.summary.histogram('weights', self.kernel)
            tf.summary.histogram('binary_weights', kernel)
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

        if self.use_act and self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return outputs


class Scalu(keras.layers.Layer):
    def __init__(self):
        super(Scalu, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_variable(
            'scale',
            shape=[1],
            initializer=tf.keras.initializers.Constant(value=0.001))

    def call(self, input):
        return input * self.scale


class QAdd(keras.layers.Layer):
    def __init__(self):
        super(QAdd, self).__init__()
        self.scope = Config.current
        self.bits = self.scope.bits
        self.act = self.scope.actQ
        self.use_q = self.scope.use_qadd

    def build(self, input_shape):
        self.scale = self.add_variable('scale', shape=[1], initializer='ones')

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        if self.use_q:
            x = self.act(x, self.bits)
            y = self.act(y, self.bits)
            output = AP2(self.scale) * (x + y)
        else:
            output = x + y
        return output


class Scale(keras.layers.Layer):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scope = Config.current
        self.use_bn = self.scope.use_bn
        self.scale = scale

    def call(self, inputs):
        if self.use_bn:
            return inputs
        else:
            return self.scale * inputs


def BatchNormalization(*args, **kwargs):
    scope = Config.current
    if scope.use_bn:
        return keras.layers.BatchNormalization(*args, **kwargs)
    else:
        #return lambda x, training: x
        return ShiftNormalization(scope, *args, **kwargs)


def MaxPool2D(*args, **kwargs):
    scope = Config.current
    if scope.use_maxpool:
        return keras.layers.MaxPool2D(*args, **kwargs)
    else:
        return lambda x: x


NormalDense = keras.layers.Dense
NormalConv2D = keras.layers.Conv2D
NormalMaxPool2D = keras.layers.MaxPool2D
NormalBatchNormalization = keras.layers.BatchNormalization
