import os
import tensorflow as tf
from .. import binary_layers as nn
#from tensorflow.keras.models import Sequential
from riptide.utils.sequential import forward_layer_list


def _conv3x3(channels, stride, normal=False):
    if normal:
        return nn.NormalConv2D(
            channels,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False)
    else:
        return nn.Conv2D(
            channels,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False)


class CIFARBasicBlockV1(tf.keras.Model):
    """BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    """

    def __init__(self,
                 channels,
                 stride,
                 downsample=False,
                 first=False,
                 **kwargs):
        super(CIFARBasicBlockV1, self).__init__(**kwargs)
        self.body = []
        if downsample:
            self.downsample = []
        else:
            self.downsample = None

        self.body.append(_conv3x3(channels, stride, normal=first))
        self.body.append(_conv3x3(channels, 1))
        if self.downsample is not None:
            self.downsample.append(
                nn.Conv2D(
                    channels, kernel_size=1, strides=stride, use_bias=False))

    def call(self, x):
        residual = x

        x = forward_layer_list(x, self.body)
        if self.downsample is not None:
            residual = forward_layer_list(residual, self.downsample)

        x = residual + x

        return x


class CIFARBasicBlockV2(tf.keras.Model):
    """BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self,
                 channels,
                 stride,
                 downsample=False,
                 first=False,
                 **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.body = []
        if downsample:
            self.downsample = []
        else:
            self.downsample = None

        self.body.append(_conv3x3(channels, stride, normal=first))
        self.body.append(_conv3x3(channels, 1))
        if self.downsample is not None:
            self.downsample.append(
                nn.Conv2D(
                    channels, kernel_size=1, strides=stride, use_bias=False))

    def call(self, x):
        residual = x

        x = forward_layer_list(x, self.body)

        if self.downsample is not None:
            residual = forward_layer_list(residual, self.downsample)

        return x + residual


class CIFARResNetV1(tf.keras.Model):
    """ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are CIFARBasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 10
        Number of classification classes.
    """

    def __init__(self, block, layers, channels, classes=10, **kwargs):
        super(CIFARResNetV1, self).__init__(**kwargs)
        with tf.name_scope("CIFARResNetV1"):
            self.features = []
            self.features.append(
                nn.NormalConv2D(
                    channels[0],
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.append(
                    self._make_layer(
                        block,
                        num_layer,
                        channels[i + 1],
                        stride,
                        i + 1,
                        in_channels=channels[i]))
            self.features.append(nn.GlobalAveragePooling2D())

            self.output_layer = []
            self.output_layer.append(nn.Dense(classes))
            self.output_layer.append(nn.Scalu())

    def _make_layer(self,
                    block,
                    layers,
                    channels,
                    stride,
                    stage_index,
                    in_channels=0):
        layer = []
        layer.append(block(channels, stride, channels != in_channels))
        for _ in range(layers - 1):
            layer.append(block(channels, 1, False))
        return layer

    def call(self, x):
        x = forward_layer_list(x, self.features)
        x = forward_layer_list(x, self.output_layer)

        return x


class CIFARResNetV2(tf.keras.Model):
    """ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are CIFARBasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 10
        Number of classification classes.
    """

    def __init__(self, block, layers, channels, classes=10, **kwargs):
        super(CIFARResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1

        self.features = []

        in_channels = channels[0]
        first = True
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.append(
                self._make_layer(
                    block,
                    num_layer,
                    channels[i + 1],
                    stride,
                    i + 1,
                    in_channels=in_channels,
                    first=first))
            first = False
            in_channels = channels[i + 1]
        self.features.append(nn.GlobalAveragePooling2D())
        self.features.append(nn.Flatten())

        self.output_layer = []
        self.output_layer.append(nn.Dense(classes))
        self.output_layer.append(nn.Scalu())

    def _make_layer(self,
                    block,
                    layers,
                    channels,
                    stride,
                    stage_index,
                    in_channels=0,
                    first=False):
        layer = []
        layer.append(
            block(channels, stride, channels != in_channels, first=first))
        for _ in range(layers - 1):
            layer.append(block(channels, 1, False))
        return layer

    def call(self, x):
        x = forward_layer_list(x, self.features)
        x = forward_layer_list(x, self.output_layer)

        return x


# Specification
resnet_net_versions = [CIFARResNetV1, CIFARResNetV2]
resnet_block_versions = [CIFARBasicBlockV1, CIFARBasicBlockV2]


def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [16, 16, 32, 64]
    layers = [n] * (len(channels) - 1)
    return layers, channels


# Constructor
def get_cifar_resnet(version, num_layers, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    """
    layers, channels = _get_resnet_spec(num_layers)

    resnet_class = resnet_net_versions[version - 1]
    block_class = resnet_block_versions[version - 1]
    net = resnet_class(block_class, layers, channels, **kwargs)

    return net


def cifar_resnet20_v1(**kwargs):
    r"""ResNet-20 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(1, 20, **kwargs)


def cifar_resnet56_v1(**kwargs):
    r"""ResNet-56 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(1, 56, **kwargs)


def cifar_resnet110_v1(**kwargs):
    r"""ResNet-110 V1 model for CIFAR10 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(1, 110, **kwargs)


def cifar_resnet20_v2(**kwargs):
    r"""ResNet-20 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(2, 20, **kwargs)


def cifar_resnet56_v2(**kwargs):
    r"""ResNet-56 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(2, 56, **kwargs)


def cifar_resnet110_v2(**kwargs):
    r"""ResNet-110 V2 model for CIFAR10 from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnet(2, 110, **kwargs)
