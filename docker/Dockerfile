#FROM jwfromm/tf-gpu
#FROM tensorflow/tensorflow:1.13.1-gpu-py3
#FROM tensorflow/tensorflow:2.0.0a0-gpu-py3
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        vim \
        git \
        curl \
        zip \
        unzip \
        pkg-config \
        software-properties-common \
        rsync \
        x11-apps \
        graphviz \
        libboost-python-dev \
        libtinfo-dev \
        libopenblas-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        llvm-dev \
        wget \
        zlib1g-dev \
	python3 \
	python3-pip \
	python-setuptools \
        && \
    rm -rf /var/lib/apt/lists/*

# Need to redefine python to proper version.
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install TVM dependencies
WORKDIR /root
RUN pip install --upgrade pip setuptools
RUN pip install numpy nose-timer cython decorator scipy sklearn pandas matplotlib

# install mkl
WORKDIR /tmp
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update && apt-get install -y intel-mkl-64bit-2019.0-045

# Set up gotools
WORKDIR /tmp
RUN wget https://dl.google.com/go/go1.11.1.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.11.1.linux-amd64.tar.gz
ENV PATH $PATH:/usr/local/go/bin

WORKDIR /
# Install buildifier
RUN go get github.com/bazelbuild/buildtools/buildifier
ENV PATH $PATH:/go/bin

RUN pip --no-cache-dir install \
	tensorflow-gpu==2.0.0-alpha0 \
	keras_applications\
	keras_preprocessing \
        Pillow \
        mixpanel \
        graphviz \
        pydot \
        ipython \
        pydot_ng \
        pyyaml \
        yapf \
        scikit-learn \
        scikit-image \
        gensim \
        cffi \
        opencv-python \
        bitfinex \
        gym \
        gym[atari] \
        flask \
        requests_ntlm \
        docopt \
        tensorboard \
        tensorflow-hub \
        tensorflow-probability \
        #tensorflow-data-validation \
        #tensorflow-transform \
        #tensorflow-model-analysis \
        jupyter \
        jupyterlab \
        keras \
        h5py \
        xgboost \
        psutil \
        seaborn \
        lime 

ENV HOME /home
WORKDIR /home

# Set up some convenience stuff
COPY .vimrc /home
COPY .gitconfig /home
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

ENV DISPLAY 0

# set utf-8 encoding, weird that we have to do this haha
ENV LC_ALL "C.UTF-8"

ENV PYTHONPATH /home/models/research:$PYTHONPATH
ENV PYTHONPATH /home/models/:$PYTHONPATH
ENV PYTHONPATH /home/Riptide:$PYTHONPATH
ENV PYTHONPATH /home/FoveNet:$PYTHONPATH
ENV TVM_HOME /home/tvm
ENV PYTHONPATH $TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:$PYTHONPATH

# Make home bazel workspace
RUN touch /home/WORKSPACE

# Set up our notebook config.
COPY jupyter_notebook_config.py /home/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

RUN chmod -R 757 /home

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888
