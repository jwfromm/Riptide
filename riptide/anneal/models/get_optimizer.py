import tensorflow as tf
from .learning_schedules import *


def get_optimizer(name, global_step, batch_size, num_gpus=1):
    schedules = {
        'adam': adam_piecewise,
        'sgd': sgd_piecewise,
        'cos': cosine_decay,
        'cyc': cyclic,
    }

    if 'adam' in name:
        name = 'adam'
    elif 'cos' in name:
        name = 'cos'
    elif 'cyc' in name:
        name = 'cyc'
    else:
        name = 'sgd'

    return schedules[name](global_step, batch_size, num_gpus)
