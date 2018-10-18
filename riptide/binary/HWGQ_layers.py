import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import nn

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

    clusters : Tensor
        When using HWGQ binarization, these are the possible values
        that can be used in approximation.

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
                 activation=None,
                 clusters=None):
        self.actQ = actQ if actQ else lambda _, x : x
        self.weightQ = weightQ if weightQ else lambda _, x : x
        self.clusters = clusters

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
        self.clusters = self.scope.clusters
    def call(self, inputs):
        if self.clusters is not None:
            inputs = self.actQ(inputs, clusters)
        else:
            inputs = self.actQ(inputs)
        kernel = self.weightQ(self.kernel)
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
