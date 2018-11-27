import tensorflow as tf


def learning_rate_with_smooth_decay(batch_size,
                                    batch_denom,
                                    decay_epochs,
                                    decay_rate,
                                    base_lr=0.1,
                                    warmup=False,
                                    staircase=False,
                                    warmup_epochs=5,
                                    num_images=1281167):
    """ Get a learning rate the smoothly decays as training progresses.

    Args:
        batch_size: Number of samples processed per batch.
        batch_denom: Base batch_size, used to scale down learning rate for smaller batches or scale up for large batches.
        decay_epochs: Number of epochs to decay the learning rate by a factor of decay_rate.
        decay_rate: Amount to decay learning rate each decay_epochs.
        base_lr: Starting learning rate.
        warmup: Run a 5 epoch warmup to the initial lr.
        staircase: If True, learning decay is not smooth.
        warmup_epochs: Number of epochs to increase the lr to the base_lr.
        num_images: Number of images in the dataset.
    """
    initial_learning_rate = base_lr * batch_size / batch_denom
    steps_per_epoch = num_images / batch_size

    def learning_rate_fn(global_step):
        if warmup:
            warmup_steps = int(steps_per_epoch * warmup_epochs)
            start_step = global_step - warmup_steps
        else:
            start_step = global_step

        lr = tf.train.exponential_decay(
            initial_learning_rate,
            start_step,
            steps_per_epoch * decay_epochs,
            decay_rate,
            staircase=staircase)

        if warmup:
            warmup_lr = (initial_learning_rate * tf.cast(
                global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
            return tf.cond(global_step < warmup_steps, lambda: warmup_lr,
                           lambda: lr)
        return lr

    return learning_rate_fn
