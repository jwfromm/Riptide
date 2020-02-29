<div align="center">
  <img src="https://www.jwfromm.com/images/Riptide-logo.png" width="400">
</div>

Riptide is a collection of functions, scripts, and tools that enable ultra
low-bitwidth neural networks to be easily trained and deployed at high speeds.
Riptide is built on top of [Tensorflow](tensorflow.org) for training and
[TVM](tvm.ai) for deployment. Riptide models uses a novel operator called *Fused
Glue* to replace all floating point operations inside of a binary neural
network. By combining Fused Glue layers with other optimizations such as
*Bitpack Fusion*, Riptide is able to generate models that run 4-12X faster than
floating point equivalents on the Raspberry Pi. For full implementation details
see our paper, which was presented at [MLSys
2020](https://mlsys.org/Conferences/2020/Schedule) and can be found [here](https://www.jwfromm.com/documents/Riptide.pdf).

# Getting Started
## Requirements
Riptide works best with Tensorflow 2.1. If you don't have tensorflow,
you can install it with `pip install tensorflow`.

Next, we can install the prerequisites to TVM
```
sudo apt-get update
sudo apt-get install -y python3 \\
                        python3-dev \\
                        python3-setuptools gcc  \\
                        libtinfo-dev zlib1g-dev \\
                        build-essential \\
                        cmake \\
                        libedit-dev \\
                        libxml2-dev

pip3 install --user numpy \\
                    decorator \\
                    attrs \\
                    tornado \\
                    psutil \\
                    xgboost                        
```

## Installing Riptide

First, recursively clone Riptide (which has a custom fork of TVM as a
submodule).

```
git clone --recursive git@github.com:jwfromm/riptide.git
```

Next we need to build the TVM submodule and set up our environment variables to properly detect it.

```
cd Riptide/tvm && mkdir build && cp cmake/config.cmake build && cd build
cmake ..
make -j4
export TVM_HOME={RiptideLocation}/tvm
export PYTHONPATH=$TVM_HOME/python;$TVM_HOME/topi/python:$PYTHONPATH
```

Note that if you want to compile a model for an embedded platform like the
Raspberry Pi, you'll need to install `llvm-dev` and set `USE_LLVM` to `ON` in
`config.cmake`.

Consider adding the above environment variables to your `.bashrc` to save
time later.

You should now be able to import Riptide in Python and are ready to train and
deploy a binary model!

We also provide a prebuilt docker image in `Riptide/docker` to make deployment
of Riptide across environments easier.

# Training a Binary Model
Riptide provides implementations of various binary layers and functions in
[binary_layers.py](riptide/binary/binary_layers.py)
and [binary_funcs.py](riptide/binary/binary_funcs.py) respectively. Although you don't need to examine or change these
files to train a model, they are written to be easy to read and adjust for other low-bit algorithms.
We provide a selection of binary models and their floating point equivalents in [riptide/models](riptide/models).
These include Alexnet, VGGNet, various Resnets, and SqueezeNet. You can create your own binary models by using
`BinaryConv2D`, `BinaryDense`, and `BatchNormalization` imported from
[binary_layers](riptide.binary.binary_layers.py).

To train a model, navigate to [scripts](scripts) and take a look at [train_imagenet.py](scripts/train_imagenet.py). This script provides a simple
and efficient interface for training models on the [ImageNet Dataset](image-net.org). We use [Tensorflow Datasets](https://www.tensorflow.org/datasets/api_docs/python/tfds)
to prepare and load images so you'll first need to download ImageNet and have `tfds.load` generate tfrecords.

Once ImageNet is ready, you can start a training job as follows:

```
python train_imagenet.py --model alexnet --experiment 2A1W --gpus 0,1,2,3 --binary --bits 2 --model_dir ~/models
```

This will start training an alexnet binarized with 2 bit activations and 1 bit weights on 4 GPUs.
Checkpoints and tensorboard logs will be saved to `model_dir/alexnet_2A1W`. Riptide automatically logs
quite a bit of useful information during training including binary histograms that are pretty neat.
To look at these logs run.

```
tensorboard --logdir ~/models/
```

Then open a browser and navigate to [localhost:6006](localhost:6006).

Training should take somewhere between a day or two to a few weeks depending on your model and number
of GPUs. Once finished, you can load the trained model as follows:

First recreate the model architecture.
```
import tensorflow as tf
from riptide.binary.binary_funcs import *
from riptide.binary.binary_layers import Config
from riptide.get_models import get_model
actQ = DQuantize
weightQ = XQuantize
config = Config(actQ=actQ, weightQ=weightQ, bits=2.0)
with config:
  model = get_model('alexnet')
```

Then we can load the checkpoint weights after initializing shapes.
```
dummy_in = tf.keras.layers.Input(shape=[224, 224, 3], batch_size=1)
dummy_out = model(dummy_in)
model.load_weights('~/models/alexnet_2A2W/model.ckpt-xxxxx)
```
Where `xxxxx` is the checkpoint identifier you want to load.

# Deploying a Binary Model
Once you've trained a binary model and are ready to run it on something
like a Raspberry Pi, it's quite simple to convert the keras
model and weights into a Relay representation.
```
import tvm
from tvm import relay
mod, params = relay.frontend.from_keras(
  model, 
  shape={'input_1': [1, 224, 224, 3]}, 
  layout='NHWC')
```

Then, we can compile the relay graph for a specific hardware platform,
in this case an ARM cpu.

```
target = tvm.garget.arm_cpu("rasp3b")
with relay.build_config(opt_level=3):
  graph, lib, params = relay.build(mod, target=target, params=params)
```

The output of `relay.build` is a set of artifacts that can be used
to run our network on an ARM CPU using the TVM runtime. One simple
way to do that is through a [TVM RPC server](https://docs.tvm.ai/tutorials/frontend/deploy_model_on_rasp.html).

```
from tvm import autotvm
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime

# Export the runtime library
tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)

# Connect to the RPC server.
remote = autotvm.measure.request_remote(
  'rasp3b', 'tracker', 9191, timeout=10000)

# Upload library and prepare to run.
remote.upload(lib_fname)
rlib = remote.load_module('net.tar')
# Create TVM runtime.
module = runtime.create(graph, rlib, ctx)
# Upload model parameters.
module.set_input(**params)

# Set input and run the model
module.set_input(0, np.random.uniform(size=(1, 224, 224, 3)))
module.run()
print(module.get_output(0))
)
```

We can also easily measure runtime using a TVM `time_evaluator`.
```
ftimer = module.module.time_evaluator(
  "run", remote.cpu(), number=10, repeat=1)
prof_res = np.array(ftimer().results) * 1000  # Convert to milliseconds
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
```

# Other Useful Things
There are a bunch of potentially useful Jupyter notebooks
located in [notebooks](notebooks). Digging through some might help
find useful examples depending on what you're trying to do.

We also provide an implementation of Riptide inside of the
[LARQ](https://github.com/larq/larq) library, which is
a similar binary network training framework. If you're interested
in this implementation please see the [riptide](https://github.com/jwfromm/larq/tree/riptide) branch of our fork
and the corresponding [riptide branch of the LARQ model zoo](https://github.com/jwfromm/zoo/tree/riptide).
