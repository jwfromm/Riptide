import tensorflow as tf
import numpy as np
from riptide.binary.binary_layers import Config
from riptide.binary.float_to_binary import convert_model
from riptide.binary.binary_funcs import DQuantize, XQuantize
from riptide.models.vgg11 import vgg11

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(
    'model_path',
    '/data/jwfromm/models/vgg_dorefa_shiftnorm_scalu/model.ckpt-60000',
    'Path to model file.')


class CorrectnessTest(tf.test.TestCase):
    def test_model(self):
        # Set up model configuration
        actQ = DQuantize
        weightQ = XQuantize
        bits = 2.0
        use_act = False
        use_bn = False
        use_maxpool = False
        pure_shiftnorm = False
        config = Config(
            actQ=actQ,
            weightQ=weightQ,
            bits=bits,
            use_act=use_act,
            use_bn=use_bn,
            use_maxpool=use_maxpool,
            pure_shiftnorm=pure_shiftnorm)
        # Build model graph.
        with config:
            model = vgg11(classes=10)
        # Load model parameters
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.model_path)

        # Run test input through network and get each layers output.
        test_input = tf.ones(shape=[1, 32, 32, 3])
        layers = model(test_input, training=False, debug=True)

        # Convert layers to integer representation for comparison to
        # fast implementation.
        converted_layers = convert_model(model, layers)

        # Check each layer versus the fast implementation TODO


if __name__ == '__main__':
    tf.test.main()
