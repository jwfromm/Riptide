import tensorflow as tf


def _forward_core(x, layers):
    for l in layers:
        if isinstance(l, list):
            x = forward_layer_list(x, l)
        else:
            x = l(x)
            act_name = "%s_activations" % tf.contrib.framework.get_name_scope()
            tf.summary.histogram(act_name, x)
    return x


def forward_layer_list(x, layers):
    if isinstance(layers[0], str):
        name = layers.pop(0)
        with tf.name_scope(name):
            return _forward_core(x, layers)
    else:
        return _forward_core(x, layers)


def forward(x, layer):
    if isinstance(layer, list):
        return forward_layer_list(x, layer)
    else:
        return layer(x)    
    

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
