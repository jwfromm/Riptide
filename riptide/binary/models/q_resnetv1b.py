"""ResNetV1bs, implemented in tf Keras."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring,dangerous-default-value

import os
import tensorflow as tf
#import tensorflow.keras.layers as nn
#from tensorflow.keras.models import Sequential
from .. import HWGQ_layers as nn
from riptide.utils.sequential import Sequential


class BasicBlockV1b(tf.keras.Model):
    """ResNetV1b BasicBlockV1b
    """
    expansion = 1

    def __init__(self,
                 planes,
                 strides=1,
                 dilation=1,
                 downsample=None,
                 previous_dilation=1,
                 norm_layer=None,
                 norm_kwargs={},
                 **kwargs):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=strides,
            padding='same',
            dilation_rate=dilation,
            use_bias=False)
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=previous_dilation,
            use_bias=False)
        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


class BottleneckV1b(tf.keras.Model):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self,
                 planes,
                 strides=1,
                 dilation=1,
                 downsample=None,
                 previous_dilation=1,
                 norm_layer=None,
                 norm_kwargs={},
                 last_gamma=False,
                 **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(filters=planes, kernel_size=1, use_bias=False)
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(
            filters=planes,
            kernel_size=3,
            strides=strides,
            padding='same',
            dilation_rate=dilation,
            use_bias=False)
        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(
            filters=planes * 4, kernel_size=1, use_bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(**norm_kwargs)
        else:
            self.bn3 = norm_layer(gamma_initializer='zeros', **norm_kwargs)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


class ResNetV1b(tf.keras.Model):
    """ Pre-trained ResNetV1b Model, which preduces the strides of 8
    featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self,
                 block,
                 layers,
                 classes=1000,
                 dilated=False,
                 norm_layer=nn.BatchNormalization,
                 norm_kwargs={},
                 last_gamma=False,
                 deep_stem=False,
                 stem_width=32,
                 avg_down=False,
                 final_drop=0.0,
                 name_prefix='',
                 **kwargs):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__(name=name_prefix)
        self.norm_kwargs = norm_kwargs
        with tf.name_scope(self.name):
            if not deep_stem:
                self.conv1 = nn.NormalConv2D(
                    filters=64,
                    kernel_size=7,
                    strides=2,
                    padding='same',
                    use_bias=False)
            else:
                self.conv1 = Sequential(name='conv1')
                self.conv1.add(
                    nn.NormalConv2D(
                        filters=stem_width,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        use_bias=False))
                self.conv1.add(norm_layer(**norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(
                    nn.Conv2D(
                        filters=stem_width,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        use_bias=False))
                self.conv1.add(norm_layer(**norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(
                    nn.Conv2D(
                        filters=stem_width * 2,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        use_bias=False))
            self.bn1 = norm_layer(**norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding='same')
            self.layer1 = self._make_layer(
                1,
                block,
                64,
                layers[0],
                avg_down=avg_down,
                norm_layer=norm_layer,
                last_gamma=last_gamma)
            self.layer2 = self._make_layer(
                2,
                block,
                128,
                layers[1],
                strides=2,
                avg_down=avg_down,
                norm_layer=norm_layer,
                last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(
                    3,
                    block,
                    256,
                    layers[2],
                    strides=1,
                    dilation=2,
                    avg_down=avg_down,
                    norm_layer=norm_layer,
                    last_gamma=last_gamma)
                self.layer4 = self._make_layer(
                    4,
                    block,
                    512,
                    layers[3],
                    strides=1,
                    dilation=4,
                    avg_down=avg_down,
                    norm_layer=norm_layer,
                    last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(
                    3,
                    block,
                    256,
                    layers[2],
                    strides=2,
                    avg_down=avg_down,
                    norm_layer=norm_layer,
                    last_gamma=last_gamma)
                self.layer4 = self._make_layer(
                    4,
                    block,
                    512,
                    layers[3],
                    strides=2,
                    avg_down=avg_down,
                    norm_layer=norm_layer,
                    last_gamma=last_gamma)
            self.avgpool = nn.GlobalAveragePooling2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(units=classes)

    def _make_layer(self,
                    stage_index,
                    block,
                    planes,
                    blocks,
                    strides=1,
                    dilation=1,
                    avg_down=False,
                    norm_layer=None,
                    last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(name='down%d' % stage_index)
            if avg_down:
                if dilation == 1:
                    downsample.add(
                        nn.AveragePooling2D(
                            pool_size=strides, strides=strides,
                            padding='same'))
                else:
                    downsample.add(
                        nn.AveragePooling2D(
                            pool_size=1, strides=1, padding='same'))
                downsample.add(
                    nn.Conv2D(
                        filters=planes * block.expansion,
                        kernel_size=1,
                        strides=1,
                        use_bias=False))
                downsample.add(norm_layer(**self.norm_kwargs))
            else:
                downsample.add(
                    nn.Conv2D(
                        filters=planes * block.expansion,
                        kernel_size=1,
                        strides=strides,
                        use_bias=False))
                downsample.add(norm_layer(**self.norm_kwargs))

        layers = Sequential(name='layers%d' % stage_index)
        if dilation in (1, 2):
            layers.add(
                block(
                    planes,
                    strides,
                    dilation=1,
                    downsample=downsample,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs,
                    last_gamma=last_gamma))
        elif dilation == 4:
            layers.add(
                block(
                    planes,
                    strides,
                    dilation=2,
                    downsample=downsample,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs,
                    last_gamma=last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.add(
                block(
                    planes,
                    dilation=dilation,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                    norm_kwargs=self.norm_kwargs,
                    last_gamma=last_gamma))

        return layers

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnet18_v1b(**kwargs):
    """Constructs a ResNetV1b-18 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    """
    model = ResNetV1b(
        BasicBlockV1b, [2, 2, 2, 2], name_prefix='resnetv1b', **kwargs)
    return model


def resnet34_v1b(**kwargs):
    """Constructs a ResNetV1b-34 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.BatchNorm`;
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    """
    model = ResNetV1b(
        BasicBlockV1b, [3, 4, 6, 3], name_prefix='resnetv1b', **kwargs)
    return model


def resnet50_v1b(**kwargs):
    """Constructs a ResNetV1b-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.BatchNorm`;
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 6, 3], name_prefix='resnetv1b', **kwargs)
    return model


def resnet101_v1b(**kwargs):
    """Constructs a ResNetV1b-101 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.BatchNorm`;
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 23, 3], name_prefix='resnetv1b', **kwargs)
    return model


def resnet152_v1b(**kwargs):
    """Constructs a ResNetV1b-152 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.BatchNorm`;
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 8, 36, 3], name_prefix='resnetv1b', **kwargs)
    return model


def resnet50_v1c(**kwargs):
    """Constructs a ResNetV1c-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 6, 3],
        deep_stem=True,
        name_prefix='resnetv1c_',
        **kwargs)
    return model


def resnet101_v1c(**kwargs):
    """Constructs a ResNetV1c-101 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 23, 3],
        deep_stem=True,
        name_prefix='resnetv1c_',
        **kwargs)
    return model


def resnet152_v1c(**kwargs):
    """Constructs a ResNetV1b-152 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 8, 36, 3],
        deep_stem=True,
        name_prefix='resnetv1c_',
        **kwargs)
    return model


def resnet50_v1d(**kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 6, 3],
        deep_stem=True,
        avg_down=True,
        name_prefix='resnetv1d_',
        **kwargs)
    return model


def resnet101_v1d(**kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 23, 3],
        deep_stem=True,
        avg_down=True,
        name_prefix='resnetv1d_',
        **kwargs)
    return model


def resnet152_v1d(**kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 8, 36, 3],
        deep_stem=True,
        avg_down=True,
        name_prefix='resnetv1d_',
        **kwargs)
    return model


def resnet50_v1e(**kwargs):
    """Constructs a ResNetV1e-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 6, 3],
        deep_stem=True,
        avg_down=True,
        stem_width=64,
        name_prefix='resnetv1e_',
        **kwargs)
    return model


def resnet101_v1e(**kwargs):
    """Constructs a ResNetV1e-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 23, 3],
        deep_stem=True,
        avg_down=True,
        stem_width=64,
        name_prefix='resnetv1e_',
        **kwargs)
    return model


def resnet152_v1e(**kwargs):
    """Constructs a ResNetV1e-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 8, 36, 3],
        deep_stem=True,
        avg_down=True,
        stem_width=64,
        name_prefix='resnetv1e_',
        **kwargs)
    return model


def resnet50_v1s(**kwargs):
    """Constructs a ResNetV1s-50 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 6, 3],
        deep_stem=True,
        stem_width=64,
        name_prefix='resnetv1s_',
        **kwargs)
    return model


def resnet101_v1s(**kwargs):
    """Constructs a ResNetV1s-101 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 4, 23, 3],
        deep_stem=True,
        stem_width=64,
        name_prefix='resnetv1s_',
        **kwargs)
    return model


def resnet152_v1s(**kwargs):
    """Constructs a ResNetV1s-152 model.

    Parameters
    ----------
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yilding a stride 8 model.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.norm_layer`;
    """
    model = ResNetV1b(
        BottleneckV1b, [3, 8, 36, 3],
        deep_stem=True,
        stem_width=64,
        name_prefix='resnetv1s_',
        **kwargs)
    return model
