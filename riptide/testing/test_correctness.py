import numpy as np
from riptide.binary.binary_layers import Config
from riptide.binary.float_to_binary import convert_model
from riptide.binary.binary_funcs import DQuantize, XQuantize, get_numpy
from end2end import verify_nnvm_vgg
from end2end.verify_nnvm_vgg import *


class CorrectnessTest(tf.test.TestCase):
    def test_model(self):
        # Verify script sets up model so just import from there.

        # Run test input through network and get each layers output.
        with graph.as_default():
            test_input = np.ones(shape=[1, 224, 224, 3], dtype=np.float32)
            test_tensor = tf.convert_to_tensor(test_input)
            layers = model(test_tensor, training=False, debug=True)

            # Convert layers to integer representation for comparison to
            # fast implementation.
            converted_layers = convert_model(model, layers)

            # Check each layer versus the fast implementation TODO
            for i, layer in enumerate(converted_layers):
                if model.layers[i].name == 'conv2d' or 'binary_dense' in model.layers[i].name:
                    nnvm_output = verify_nnvm_vgg.run(
                        test_input, stop_layer=model.layers[i].name)
                    layer_np = get_numpy(sess, layer)

                    correct = np.allclose(layer_np, nnvm_output, rtol=1e-3)
                    if not correct:
                        print("Mismatch on layer %d: %s" %
                              (i, model.layers[i].name))

                elif 'binary_conv2d' in model.layers[i].name:
                    nnvm_output = verify_nnvm_vgg.run(
                        test_input, stop_layer=model.layers[i].name)
                    layer_np = get_numpy(sess, converted_layers[i + 1])

                    correct = np.allclose(layer_np, nnvm_output, rtol=1e-3)
                    if not correct:
                        print("Mismatch on layer %d: %s" %
                              (i, model.layers[i].name))


if __name__ == '__main__':
    tf.test.main()
