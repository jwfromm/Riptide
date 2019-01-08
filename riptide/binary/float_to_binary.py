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
                converted_layers.append(converted_layer)
            elif 'shift_normalization' in layer.name:
                converted_layer = DQuantizeBits(layers[i], bits=bits)
                converted_layers.append(converted_layer)
    return converted_layers
