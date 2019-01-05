import tensorflow as tf
from .binary_funcs import get_quantize_bits, get_shiftnorm_ap2, log2, DQuantizeBits


# Takes the loaded model and outputs at each layer in floating point
# and returns a list of the layers in the equivalent integer representation.
def convert_model(model, layers, bits=2.0):
    converted_layers = []
    for i, layer in enumerate(model.layers):
        if 'shift' not in layer.name and 'binary' not in layer.name:
            converted_layers.append(layers[i])
        else:
            if 'binary' in layer.name:
                mean, _ = get_quantize_bits(model.layers[i].weights[0])
                scale_factor = (bits**2 - 1) * 2**(-log2(mean))
                converted_layer = tf.round(layers[i] * scale_factor)
                converted_layer = DQuantizeBits(converted_layer, bits=bits)
                converted_layers.append(converted_layer)
            elif 'shift_normalization' in layer.name:

                # Find preceding conv layer.
                if 'max_pooling2d' in model.layers[i - 1].name:
                    layer_offset = 2
                else:
                    layer_offset = 1

                mean, _ = get_quantize_bits(
                    model.layers[i - layer_offset].weights[0])
                shift_std, shift_mean = get_shiftnorm_ap2(
                    model.layers[i],
                    conv_weights=model.layers[i - layer_offset].weights[0],
                    rescale=True)
                total_shift = -log2(mean) - log2(shift_std)
                scale_factor = (bits**2 - 1) * 2**total_shift
                converted_layer = tf.round(layers[i] * scale_factor)
                converted_layer = DQuantizeBits(converted_layer, bits=bits)
                converted_layers.append(converted_layer)
    return converted_layers
