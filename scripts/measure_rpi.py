import os
import numpy as np
import tensorflow as tf
import argparse

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
import tvm.relay.testing.tf as tf_testing
from tvm.autotvm.tuner import XGBTuner, GATuner, GridSearchTuner
from tvm.contrib.util import tempdir
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime

import riptide.models
from riptide.get_models import get_model
from riptide.binary.binary_layers import Config, DQuantize, XQuantize

os.environ["CUDA_VISIBLE_DEVICES"] = ''

device_key = 'rpi3b'
target = tvm.target.arm_cpu("rasp3b")
target_host = 'llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--activation_bits',
    type=int,
    default=1,
    help='number of activation bits',
    required=False)
parser.add_argument(
    '--model',
    type=str,
    choices=['vggnet', 'vgg11', 'resnet18', 'alexnet', 'darknet', 'squeezenet', 'squeezenet_normal'],
    help='neural network model',
    required=True)
parser.add_argument(
    '--log_file',
    type=str,
    default='log.log',
    help='logfile to store tuning results',
    required=False)
args = parser.parse_args()
model = args.model
activation_bits = args.activation_bits
log_file = args.log_file

config = Config(
    actQ=DQuantize,
    weightQ=XQuantize,
    bits=activation_bits,
    use_act=False,
    use_bn=False,
    use_maxpool=True)

with config:
    model = get_model(model)
#model = riptide.models.vggnet_normal.vggnet()

# Init model shapes.
test_input = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype='float32')
output = model(test_input)

# Parse model to relay
with target:
    net, params = relay.frontend.from_keras(
        model, shape={
            'input_1': [1, 224, 224, 3]
        }, layout='NHWC')
num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

with autotvm.apply_history_best(log_file):
        print("Compile...")

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target, params=params)

        batch_size = 1
        num_class = 1000
        image_shape = (224, 224, 3)
        data_shape = (batch_size, ) + image_shape

        tmp = util.tempdir()
        lib_fname = tmp.relpath('net.tar')
        lib.export_library(lib_fname)

        # Upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(
            device_key, 'fleet.cs.washington.edu', 9190, timeout=10000)
        # upload the library to remote device and load it
        remote.upload(lib_fname)
        rlib = remote.load_module('net.tar')

        # create the remote runtime module
        ctx = remote.cpu(0)
        module = runtime.create(graph, rlib, ctx)
        # set parameter (upload params to the remote device. This may take a while)
        module.set_input(**params)
        module.set_input(
            'input_1',
            tvm.nd.array(
                np.random.uniform(size=image_shape).astype('float32')))
        module.run()
        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=1)
        prof_res = np.array(ftimer().results) * 1000  # Convert to milliseconds
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

