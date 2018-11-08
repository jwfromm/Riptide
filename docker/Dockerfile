FROM jwfromm/tf-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        vim \
        unzip \
        software-properties-common \
        rsync \
        x11-apps \
        graphviz \
        libboost-python-dev \
        libtinfo-dev \
        libopenblas-dev \
        llvm-dev \
        yapf \
        && \
    rm -rf /var/lib/apt/lists/*

# Need to redefine python to proper version.
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Install TVM dependencies
WORKDIR /root
RUN pip install numpy nose-timer cython decorator scipy

# install mkl
WORKDIR /tmp
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update && apt-get install -y intel-mkl-64bit-2019.0-045

# install tvm
WORKDIR /root
RUN git clone --recursive https://github.com/jwfromm/tvm.git
RUN mkdir /root/tvm/build
COPY tvm_config.cmake /root/tvm/build/config.cmake
WORKDIR /root/tvm/build
RUN cmake ..
RUN make -j8
#WORKDIR /tvm/python
#RUN python setup.py install
#WORKDIR /tvm/topi/python
#RUN python setup.py install
#WORKDIR /tvm/nnvm/python
#RUN python setup.py install
ENV PYTHONPATH /root/tvm/python:$PYTHONPATH
ENV PYTHONPATH /root/tvm/topi/python:$PYTHONPATH
ENV PYTHONPATH /root/tvm/nnvm/python:$PYTHONPATH
ENV PYTHONPATH /root/tvm/vta/python:$PYTHONPATH
WORKDIR /root/tvm
RUN make cython3

# Set up gotools
WORKDIR /tmp
RUN wget https://dl.google.com/go/go1.11.1.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.11.1.linux-amd64.tar.gz
ENV PATH $PATH:/usr/local/go/bin
WORKDIR /root

# Install buildifier
RUN go get github.com/bazelbuild/buildtools/buildifier
ENV PATH $PATH:/root/go/bin

RUN pip --no-cache-dir install \
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
        mxnet

# Set up some convenience stuff
COPY .vim /root/.vim
COPY .vimrc /root
COPY .gitconfig /root

ENV DISPLAY 0

WORKDIR /workspace
WORKDIR /root
# set utf-8 encoding, weird that we have to do this haha
ENV LC_ALL "C.UTF-8"

ENV PYTHONPATH /root/models/research:$PYTHONPATH
ENV PYTHONPATH /root/models/:$PYTHONPATH
ENV PYTHONPATH /root/Riptide:$PYTHONPATH

# Make root bazel workspace
RUN touch /root/WORKSPACE

RUN chmod -R a+w /workspace
