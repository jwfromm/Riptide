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
2020](https://mlsys.org/Conferences/2020/Schedule).

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
Riptide provides implementations of various binary layers and functions in `Riptide/riptide/binary_layers.py`
and `Riptide/riptide/binary_functions.py` respectively. Although you don't need to examine or change these
files to train a model, they are written to be easy to read and adjust for other low-bit algorithms.
We provide a selection of binary models and their floating point equivalents in 'Riptide/riptide/models`.
These include Alexnet, VGGNet, various Resnets, and SqueezeNet. You can create your own binary models by using
`BinaryConv2D`, `BinaryDense`, and `BatchNormalization` imported from `riptide.binary_layers`.

# riptide
Core codebase of the repo. See riptide/binary for training implementations of various binary functions. Models contains a small
selection of binary enabled architectures.

# Notebooks
Various ipython notebooks that are useful for quick experiments and debugging. See CifarExperiments.ipynb for a simple
example of training and evaluating binary models using CIFAR10.
