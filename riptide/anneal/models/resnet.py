import six
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Activation, Reshape, Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Add
from tensorflow.keras.regularizers import l2
from riptide.anneal.anneal_funcs import *


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS

    if keras.backend.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _bn_relu(x, bn_name=None):
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return PACT()(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(x):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name)

    return f


def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name)
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    # Adds a shortcut between input and residual block and merges them with "sum".

    input_shape = input_feature.shape
    residual_shape = residual.shape
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] /
                              residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1x1 conv if shape is different, otherwise identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = BatchNormalization(axis=CHANNEL_AXIS,
                                      name=bn_name_base)(shortcut)

    return Add()([shortcut, residual])


def _residual_block(block_function,
                    filters,
                    blocks,
                    stage,
                    transition_strides=None,
                    transition_dilation_rates=None,
                    dilation_rates=None,
                    is_first_layer=False,
                    dropout=None,
                    residual_unit=_bn_relu_conv):
    """Builds a residual block with repeating bottleneck blocks.
        stage: integer, current stage label, used for generating label names.
        blocks: number of blocks 'a', 'b', ... current block label used for generating
            names.
        transition_strides: a list of tuples for strides of each transition.
        transition_dilation_rates: a list of tuples for the dilation rate of each transition.
    """
    if transition_dilation_rates is None:
        transition_dilation_rates = [(1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters,
                               stage=stage,
                               block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
            return x

    return f


def _block_name_base(stage, block):
    # Get the convolution and batchnorm name for this stage and block.
    if block < 27:
        block = '%c' % (block + 97)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base


def basic_block(filters,
                stage,
                block,
                transition_strides=(1, 1),
                dilation_rate=(1, 1),
                is_first_block_of_first_layer=False,
                dropout=None,
                residual_unit=_bn_relu_conv):
    # Basic 3 x 3 convolution blocks for resnets with fewer than 34 layers.
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # Dont repeat bn-> relu since we just did bn->relu->maxpool (also dont binarize)
            x = Conv2D(filters=filters,
                       kernel_size=(3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters,
                              kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)
        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters,
                          kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)

    return f


def bottleneck(filters,
               stage,
               block,
               transition_strides=(1, 1),
               dilation_rate=(1, 1),
               is_first_block_of_first_layer=False,
               dropout=None,
               residual_unit=_bn_relu_conv):
    # Bottleneck architecture for > 34 layer resnet.
    def f(input_feature):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # Dont repeat bn->relu since we just did bn->relu->maxpool also dont quantize.
            x = Conv2D(filters=filters,
                       kernel_size=(1, 1),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_feature)
        else:
            x = residual_unit(filters=filters,
                              kernel_size=(1, 1),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_feature)

        if dropout is not None:
            x = Dropout(dropout(x))

        x = residual_unit(filters=filters * 4,
                          kernel_size=(1, 1),
                          conv_name_base=conv_name_base + '2c',
                          bn_name_base=bn_name_base + '2c')(x)

        return _shortcut(input_feature, x)

    return f


def _string_to_function(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def ResNet(input_shape=None,
           classes=1000,
           block='bottleneck',
           residual_unit='v1',
           reptitions=None,
           initial_filters=64,
           activation='softmax',
           include_top=True,
           input_tensor=None,
           dropout=None,
           transition_dilation_rate=(1, 1),
           initial_strides=(2, 2),
           initial_kernel_size=(7, 7),
           initial_pooling='max',
           final_pooling=None,
           top='classification'):
    # Builds a resnet architecture

    if reptitions is None:
        reptitions = [3, 4, 6, 3]

    _handle_dim_ordering()

    if input_shape is None:
        input_shape = [224, 224, 3]

    if block == 'basic':
        block_fn = basic_block
    elif block == 'bottleneck':
        block_fn = bottleneck
    elif isinstance(block, six.string_types):
        block_fn = _string_to_function(block)
    else:
        block_fn = block

    if residual_unit == 'v2':
        residual_unit = _bn_relu_conv
    elif residual_unit == 'v1':
        residual_unit = _conv_bn_relu
    elif isinstance(residual_unit, six.string_types):
        residual_unit = _string_to_function(residual_unit)
    else:
        residual_unit = residual_unit

    img_input = Input(shape=input_shape, tensor=input_tensor)
    x = _conv_bn_relu(filters=initial_filters,
                      kernel_size=initial_kernel_size,
                      strides=initial_strides)(img_input)
    if initial_pooling == 'max':
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=initial_strides,
                         padding="same")(x)

    block = x
    filters = initial_filters
    for i, r in enumerate(reptitions):
        transition_dilation_rates = [transition_dilation_rate * r]
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(
            block_fn,
            filters=filters,
            stage=i,
            blocks=r,
            is_first_layer=(i == 0),
            dropout=dropout,
            transition_dilation_rates=transition_dilation_rates,
            transition_strides=transition_strides,
            residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)

    # Classifier block

    if include_top and top is 'classification':
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=classes,
                  activation=activation,
                  kernel_initializer="he_normal")(x)

    model = keras.Model(inputs=img_input, outputs=x)
    return model


def ResNet18(input_shape=[224, 224, 3], classes=1000):
    return ResNet(input_shape, classes, basic_block, reptitions=[2, 2, 2, 2])
