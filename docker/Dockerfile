FROM jwfromm/tf-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        vim \
        unzip \
        libboost-python-dev \
        software-properties-common \
        rsync \
        x11-apps \
        graphviz \
        libtinfo-dev \
        python-setuptools \
        libopenblas-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install TVM
WORKDIR /root
RUN pip install numpy nose-timer cython decorator scipy

# LLVM
RUN echo deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main \
     >> /etc/apt/sources.list.d/llvm.list && \
     wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - && \
     apt-get update && apt-get install -y --force-yes llvm-6.0

RUN git clone https://github.com/dmlc/tvm --recursive
WORKDIR tvm
RUN mkdir build
COPY config.cmake build/config.cmake
WORKDIR build
RUN cmake ..
RUN make -j10
ENV PYTHONPATH=/tvm/python:/tvm/topi/python:/tvm/nnvm/python/:/tvm/vta/python:${PYTHONPATH}

# clone models
WORKDIR /
RUN git clone --recursive https://github.com/jwfromm/models.git

RUN pip --no-cache-dir install \
        Pillow \
        ipykernel \
        mixpanel \
        graphviz \
        pydot \
        pydot_ng \
        pyyaml \
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
        tensorflow-data-validation \
        tensorflow-transform \
        tensorflow-model-analysis \
        jupyterlab \
        keras \
        h5py \
        && \
    python -m ipykernel.kernelspec

# Set up some convenience stuff
COPY .vimrc /root
COPY .gitconfig /root
COPY .vim /root/.vim

ENV DISPLAY 0

WORKDIR /workspace
WORKDIR /root
# set utf-8 encoding, weird that we have to do this haha
ENV LC_ALL "C.UTF-8"
ENV PYTHONPATH /models/research:$PYTHONPATH

RUN chmod -R a+w /workspace
