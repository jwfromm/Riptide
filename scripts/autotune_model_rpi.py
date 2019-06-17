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

log_file = "rpi3_quantized.log"
device_key = 'rpi3b'
target = tvm.target.arm_cpu("rasp3b")
target_host = 'llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon'
ctx = tvm.cpu(0)

config = Config(actQ=DQuantize, weightQ=XQuantize, bits=1, use_act=False, use_bn=False, use_maxpool=True)

with config:
    model = get_model('vggnet')
#model = riptide.models.vggnet_normal.vggnet()

# Init model shapes.
input_shape = [1, 3, 224, 224]
test_input = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype='float32')
output = model(test_input)

# Parse model to relay
with target:
    net, params = relay.frontend.from_keras(model, shape={'input_1': [1, 224, 224, 3]})

num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

# Set up tuning options

tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'early_stopping': None,
    'n_trial': 200,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func='default'),
        runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9190,
            number=5, timeout=10)
    ),
}


def tune_kernels(tasks,
                 measure_option,
                 tuner='xgb',
                 n_trial=100,
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # Converting conv2d tasks to conv2d_NCHWc tasks. Do we actually want this?
        op_name = tsk.workload[0]
        input_shape = tsk.workload[1][0:-1]
        kernel_shape = tsk.workload[2][0:-1]
        input_channels = input_shape[1]
        # Only can convert to NCHWc if input channels is divisible by 8.
        #convertible = input_channels % 8 == 0
        func_create = tsk.name
        #if op_name == 'conv2d':
        #    func_create = 'topi_x86_conv2d_NCHWc'
        #elif op_name == 'depthwise_conv2d_nchw':
        #    func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'

        print(func_create)
        task = autotvm.task.create(func_create, args=tsk.args, target=target, template_key='direct')

        task.workload = tsk.workload

        # Create tuner.
        if tuner == 'xgb' or tuner == 'xbg-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tunder: " + tuner)

        # Do tuning.
        n_trial = min(n_trial, len(task.config_space))
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])

# Launch jobs and evaluate performance.
def tune_and_evaluate(tuning_opt):
    print("Extract tasks...")
    global net, params, input_shape
    tasks = autotvm.task.extract_from_program(
        net,
        target=target,
        params=params,
        ops=(relay.op.nn.conv2d, relay.op.nn.dense, relay.op.nn.bitserial_conv2d,
             relay.op.nn.bitserial_dense))

    # Run tuning tasks.
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)

#    # compile kernels with historgy best records.
#    with autotvm.apply_history_best(log_file):
#        print("Compile...")
#        with relay.build_config(opt_level=3):
#            graph, lib, params = relay.build_module.build(
#                net, target=target, params=params)
#
#        # Export library
#        tmp = tempdir()
#        filename = 'net.so'
#        lib.export_library(tmp.relpath(filename))
#
#        # Upload module to device
#        print("Upload...")
#        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190, timeout=10000)
#
#        remote.upload(tmp.relpath(filename))
#        rlib = remote.load_module(filename)
#
#        # upload parameters to device
#        ctx = remote.context(str(target), 0)
#        module = runtime.create(graph, rlib, ctx)
#        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
#        module.set_input('input_1', data_tvm)
#        module.set_input(**params)
#
#        # evaluate
#        print("Evaluate inference time cost...")
#        ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=1)
#        prof_res = np.array(ftimer().results) * 1000 # Convert to milliseconds
#        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
#              (np.mean(prof_res), np.std(prof_res)))

tune_and_evaluate(tuning_option)
