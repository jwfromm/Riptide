import os
import numpy as np
import tensorflow as tf

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
import tvm.relay.testing.tf as tf_testing
from tvm.autotvm.tuner import XGBTuner, GATuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

import riptide.models
from riptide.get_models import get_model
from riptide.binary.binary_layers import Config, DQuantize, XQuantize

os.environ["CUDA_VISIBLE_DEVICES"] = ''

log_file = "rpi_test.log"
device_key = 'rpi3b'
target = tvm.target.arm_cpu("rasp3b")
ctx = tvm.cpu(0)

config = Config(actQ=DQuantize, weightQ=XQuantize, bits=1, use_act=False, use_bn=False, use_maxpool=True, bipolar=True)

with config:
    model = get_model('vggnet')
#model = riptide.models.vggnet_normal.vggnet()

# Init model shapes.
input_shape = [1, 64, 64, 64]
test_input = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype='float32')
output = model(test_input)

# Parse model to relay
with target:
    net, params = relay.frontend.from_keras(model, shape={'input_1': [1, 64, 64, 64]}, layout='NHWC')

num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

# compile kernels with historgy best records.
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            net, target=target, params=params)

    # Export library
    tmp = tempdir()
    filename = 'net.tar'
    #lib.export_library(tmp.relpath(filename), cc="/usr/bin/arm-linux-gnueabihf-g++")
    lib.export_library(tmp.relpath(filename))

    # Upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(
            device_key, 'fleet.cs.washington.edu', 9190, timeout=10000)

    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
    module.set_input('input_1', data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=1)
    prof_res = np.array(ftimer().results) * 1000 # Convert to milliseconds
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))

