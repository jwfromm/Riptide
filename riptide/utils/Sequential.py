import tensorflow as tf


class Sequential(tf.keras.Model):
    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__()
        self.layer_list = []
        if layers is not None:
            self.layers.append(layers)
        self.name_scope = name

    def add(self, l):
        self.layer_list.append(l)

    def call(self, x):
        with tf.name_scope(self.name_scope):
            for l in self.layer_list:
                if isinstance(l, list):
                    x = self.call(l)
                else:
                    x = l(x)
        return x
