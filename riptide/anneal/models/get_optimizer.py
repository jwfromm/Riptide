import tensorflow as tf
from .learning_schedules import *

_NUM_IMAGES = 1281167


def get_optimizer(name, global_step, batch_size, num_gpus=1):
    models = {
        'alexnet': adam_piecewise,
        'alexnet_sgd': sgd_piecewise,
        'alexnet_cos': cosine_decay,
        'alexnet_cyc': cyclic,
    }

    return models[name](global_step, batch_size, num_gpus)
