import tensorflow as tf
from .clr import cyclic_learning_rate

_NUM_IMAGES = 1281167


def adjust_start_lr(lr, batch_size, num_gpus, batch_denom=128):
    if num_gpus == 0:
        num_gpus = 1

    batch_size = batch_size * num_gpus

    lr_adjustment = batch_size / batch_denom
    starting_lr = lr * lr_adjustment
    return starting_lr, _NUM_IMAGES / batch_size


def adam_piecewise(global_step, batch_size, num_gpus):
    starting_lr, steps_per_epoch = adjust_start_lr(1e-4, batch_size, num_gpus)

    lr_decay = 0.2
    lr_values = [
        starting_lr, starting_lr * lr_decay, starting_lr * lr_decay * lr_decay
    ]
    epoch_boundaries = [56, 64]
    step_boundaries = [int(x * steps_per_epoch) for x in epoch_boundaries]
    lr_schedule = tf.compat.v1.train.piecewise_constant_decay(
        global_step, boundaries=step_boundaries, values=lr_values)
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr_schedule, epsilon=1e-5)
    return optimizer, lr_schedule


def sgd_piecewise(global_step, batch_size, num_gpus):
    starting_lr, steps_per_epoch = adjust_start_lr(0.1, batch_size, num_gpus, batch_denom=256)

    lr_values = [0.1, 1e-2, 1e-3, 1e-4, 1e-5]
    epoch_boundaries = [30, 60, 85, 95]
    step_boundaries = [int(x * steps_per_epoch) for x in epoch_boundaries]
    lr_schedule = tf.compat.v1.train.piecewise_constant_decay(
        global_step, boundaries=step_boundaries, values=lr_values)
    optimizer = tf.compat.v1.train.MomentumOptimizer(lr_schedule, 0.9)

    return optimizer, lr_schedule


def cosine_decay(global_step, batch_size, num_gpus):
    starting_lr, steps_per_epoch = adjust_start_lr(.0128, batch_size, num_gpus, batch_denom=128)

    lr_schedule = tf.compat.v1.train.cosine_decay_restarts(starting_lr, global_step, 1000)

    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=lr_schedule,
        momentum=0.9,
        use_nesterov=False)

    return optimizer, lr_schedule

def cyclic(global_step, batch_size, num_gpus):
    max_lr, steps_per_epoch = adjust_start_lr(.1, batch_size, num_gpus, batch_denom=128)

    step_size = 4 * steps_per_epoch

    lr_schedule = cyclic_learning_rate(
        global_step, learning_rate=1e-6,
        max_lr=max_lr, step_size=step_size, mode='triangular')

    # Momentum should vary inversely to learning rate.
    max_momentum = 0.9
    min_momentum = 0.0
    momentum_schedule = max_momentum - cyclic_learning_rate(
        global_step, learning_rate=min_momentum,
        max_lr=max_momentum, step_size=step_size, mode='triangular')

    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=lr_schedule,
        momentum=momentum_schedule,
        use_nesterov=False)
    return optimizer, lr_schedule
