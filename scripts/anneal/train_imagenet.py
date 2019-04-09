import os
import multiprocessing
import tensorflow as tf
from functools import partial
from riptide.utils.datasets import imagerecord_dataset
from riptide.utils.thread_helper import setup_gpu_threadpool
from riptide.anneal.anneal_funcs import SAWBConv2D, PACT
from riptide.anneal.anneal_config import Config
from riptide.anneal.models import get_model, get_optimizer
from slim.preprocessing.inception_preprocessing import preprocess_image

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model_dir', '/data/jwfromm/anneal_models',
                       'Directory to save models in.')
tf.flags.DEFINE_string('model', '', 'Name of model to train, must be set.')
tf.flags.DEFINE_string(
    'experiment', '',
    'Suffix to add to model name, should describe purpose of run.')
tf.flags.DEFINE_string('data_path', '/data/imagenet/tfrecords',
                       'Directory containing tfrecords to load.')
tf.flags.DEFINE_string('gpus', '', 'Comma seperated list of GPUS to run on.')
tf.flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
tf.flags.DEFINE_integer('batch_size', 64, 'Size of each minibatch.')
tf.flags.DEFINE_integer('image_size', 224,
                        'Height and Width of processed images.')
tf.flags.DEFINE_float('learning_rate', .0128, 'Starting learning rate.')
tf.flags.DEFINE_float('momentum', 0.9, 'Momentum used for optimizer.')
tf.flags.DEFINE_boolean('quantize', 0, 'Use a quantized network.')
tf.flags.DEFINE_float('a_bits', 2.0,
                      'Number of activation bits to use for binary model.')
tf.flags.DEFINE_float('w_bits', 2.0,
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
        return imagerecord_dataset(
            FLAGS.data_path,
            FLAGS.batch_size,
            is_training=False,
            preprocess=eval_preprocess,
            num_workers=num_workers)

    # Set up estimaor model function.
    def model_fn(features, labels, mode):
        # Generate summary for input images.
        tf.summary.image('images', features, max_outputs=4)

        if FLAGS.quantize:
            a_bits = FLAGS.a_bits
            w_bits = FLAGS.w_bits
            quantize = True
        else:
            a_bits = None
            w_bits = None
            quantize = False

        config = Config(quantize=quantize, a_bits=a_bits, w_bits=w_bits)

        with config:
            model = get_model(FLAGS.model)

        # Get proper mode for batchnorm and dropout, must be python bool.
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False

        logits = model(features, training=is_training)
        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        # Calcuate loss for train and eval modes.
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        model_losses = model.get_losses_for(features) + model.get_losses_for(
            None)
        # Compute summed loss.
        loss = cross_entropy
        if model_losses:
            loss += tf.math.add_n(model_losses)
        # Log model losses.
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Compute training metrics.
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
        accuracy_top_5 = tf.metrics.mean(
            tf.nn.in_top_k(
                predictions=logits,
                targets=tf.reshape(labels, [-1]),
                k=5,
                name='top_5_op'))
        metrics = {'accuracy': accuracy, 'accuracy_top_5': accuracy_top_5}

        # Ready to configure the EVAL mode specification.
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=metrics)

        # Otherwise, we must be doing training.
        global_step = tf.train.get_or_create_global_step()
        optimizer, learning_rate = get_optimizer(FLAGS.model, global_step,
                                                 FLAGS.batch_size)
        # Track learning rate.
        tf.summary.scalar('learning_rate', learning_rate)
        update_ops = model.get_updates_for(features) + model.get_updates_for(
            None)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)
        # Keep track of training accuracy.
        tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    # Now we're ready to configure our estimator and train.
    # Determine proper name for this model.
    full_model_path = os.path.join(FLAGS.model_dir,
                                   "%s_%s" % (FLAGS.model, FLAGS.experiment))
    # Figure out which GPUS to run on.
    if num_gpus > 1:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        strategy = None
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=op_threads,
        intra_op_parallelism_threads=op_threads,
        allow_soft_placement=True)
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
    tf.app.run()
