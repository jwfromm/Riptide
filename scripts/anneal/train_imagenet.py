import os
import multiprocessing
import tensorflow as tf
from functools import partial
from riptide.utils.datasets import imagerecord_dataset
from riptide.utils.thread_helper import setup_gpu_threadpool
from riptide.anneal.anneal_config import Config
from riptide.anneal.models import get_model, get_optimizer
from riptide.utils.preprocessing.inception_preprocessing import preprocess_image

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/data/jwfromm/anneal_models',
                    'Directory to save models in.')
flags.DEFINE_string('model', '', 'Name of model to train, must be set.')
flags.DEFINE_string(
    'experiment', '',
    'Suffix to add to model name, should describe purpose of run.')
flags.DEFINE_string('data_path', '/data/imagenet/tfrecords',
                    'Directory containing tfrecords to load.')
flags.DEFINE_string('gpus', '', 'Comma seperated list of GPUS to run on.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 64, 'Size of each minibatch.')
flags.DEFINE_integer('image_size', 224,
                     'Height and Width of processed images.')
flags.DEFINE_float('learning_rate', .0128, 'Starting learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum used for optimizer.')
flags.DEFINE_boolean('quantize', 0, 'Use a quantized network.')
flags.DEFINE_float('a_bits', 2.0,
                   'Number of activation bits to use for binary model.')
flags.DEFINE_float('w_bits', 2.0,
                   'Number of activation bits to use for binary model.')


def main(argv):
    # Set visible GPUS appropriately.
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    # Get thread confirguration.
    op_threads, num_workers = setup_gpu_threadpool(len(FLAGS.gpus.split(',')))
    num_gpus = len(FLAGS.gpus.split(','))
    # Set up the data input functions.
    train_preprocess = partial(
        preprocess_image,
        height=FLAGS.image_size,
        width=FLAGS.image_size,
        is_training=True)
    eval_preprocess = partial(
        preprocess_image,
        height=FLAGS.image_size,
        width=FLAGS.image_size,
        is_training=False)

    def train_input_fn():
        ds = imagerecord_dataset(
            FLAGS.data_path,
            FLAGS.batch_size,
            is_training=True,
            preprocess=train_preprocess,
            num_workers=num_workers)
        return ds.repeat(FLAGS.epochs)

    def eval_input_fn():
        ds = imagerecord_dataset(
            FLAGS.data_path,
            FLAGS.batch_size,
            is_training=False,
            preprocess=eval_preprocess,
            num_workers=num_workers)
        return ds.repeat(1)

    # Set up estimaor model function.
    def model_fn(features, labels, mode):
        # Generate summary for input images.
        tf.compat.v1.summary.image('images', features, max_outputs=4)

        if FLAGS.quantize:
            a_bits = FLAGS.a_bits
            w_bits = FLAGS.w_bits
            quantize = True
            fixed = False# Do standard dorefa quantization
        else:
            a_bits = None
            w_bits = None
            quantize = False
            fixed = False

        config = Config(quantize=quantize, a_bits=a_bits, w_bits=w_bits, fixed=fixed)

        with config:
            model = get_model(FLAGS.model)

        global_step = tf.compat.v1.train.get_or_create_global_step()
        optimizer, learning_rate = get_optimizer(FLAGS.model, global_step,
                                                 FLAGS.batch_size, num_gpus)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        def loss_fn(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=FLAGS.batch_size)

        # Track learning rate.
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        # Get proper mode for batchnorm and dropout, must be python bool.
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        predictions = model(features, training=training)

        total_loss = loss_fn(labels, predictions)
        reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
        if reg_losses:
            total_loss += tf.math.add_n(reg_losses)

        # Compute training metrics.
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=labels,
            predictions=tf.math.argmax(predictions, axis=-1),
            name='acc_op')
        accuracy_top_5 = tf.compat.v1.metrics.mean(
            tf.math.in_top_k(
                predictions=predictions,
                targets=tf.reshape(labels, [-1]),
                k=5,
                name='top_5_op'))
        metrics = {'accuracy': accuracy, 'accuracy_top_5': accuracy_top_5}

        update_ops = model.get_updates_for(features) + model.get_updates_for(
            None)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                total_loss,
                var_list=model.trainable_variables,
                global_step=global_step)
        # Keep track of training accuracy.
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
            tf.compat.v1.summary.scalar('train_accuracy_top_5',
                                        accuracy_top_5[1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    # Now we're ready to configure our estimator and train.
    # Determine proper name for this model.
    full_model_path = os.path.join(FLAGS.model_dir,
                                   "%s_%s" % (FLAGS.model, FLAGS.experiment))
    # Figure out which GPUS to run on.
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    session_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=op_threads,
        intra_op_parallelism_threads=op_threads,
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_summary_steps=500,
        log_step_count_steps=500,
        save_checkpoints_secs=3600,
        train_distribute=strategy,
        session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=full_model_path, config=run_config)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    app.run(main)
