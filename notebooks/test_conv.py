import tvm
from tvm import relay
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime
from tvm.contrib import util
from tvm.contrib.util import tempdir
import numpy as np

target = tvm.target.arm_cpu("rasp3b")

x = relay.var('x', shape=[1, 224, 224, 32], dtype='int16')
w = relay.var('w', shape=[1, 1, 32, 32], dtype='int16')
w_np = np.random.normal(size=[1, 1, 32, 32]).astype('int16')
x_np = np.random.normal(size=[1, 224, 224, 32]).astype('int16')

y = relay.nn.conv2d(x, w, data_layout='NHWC', kernel_size=[1, 1], channels=32, kernel_layout='HWIO')
y_func = relay.Function([x, w], y)

params = {'w': w_np}
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(y_func, target=target, params=params)

tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)

remote = autotvm.measure.request_remote(
    'rpi3b', 'fleet.cs.washington.edu', 9190, timeout=10000)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module('net.tar')

# create the remote runtime module
ctx = remote.cpu(0)
module = runtime.create(graph, rlib, ctx)
# set parameter (upload params to the remote device. This may take a while)
module.set_input(**params)

module.set_input('x', x_np)
module.run()

 # Evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=1)
prof_res = np.array(ftimer().results) * 1000  # Convert to milliseconds
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
