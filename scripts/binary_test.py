import os
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.client import timeline
from riptide.binary.binary_ops import binary_dense
from tensorflow.contrib.compiler import jit

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool('binary', False,
        'Whether to run with real binary ops or not.')
tf.flags.DEFINE_integer('neurons', 512,
        'Number of neurons in dense layer.')
tf.flags.DEFINE_bool('xla', False,
        'Whether to compile with xla or not.')
tf.flags.DEFINE_integer('batch_size', 1,
        'Number of samples per batch.')

# Disable GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''


with tf.Session() as sess:
    jit_scope = jit.experimental_jit_scope

    with jit_scope(compile_ops=FLAGS.xla):
        if FLAGS.binary:
            a = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, FLAGS.neurons])
            b = tf.placeholder(tf.int64, shape=[FLAGS.neurons/64, FLAGS.neurons])
            test_op = binary_dense(a, b, binarize_a=True, binarize_b=False)
            test_op = tf.expand_dims(test_op, axis=0)
        else:
            a = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.neurons])
            b = tf.placeholder(tf.float32, shape=[FLAGS.neurons, FLAGS.neurons])
            test_op = tf.matmul(tf.sign(a), tf.sign(b))

    a_np = np.random.normal(size=a.shape.as_list())
    b_np = np.random.normal(size=b.shape.as_list())

    #sess.run(tf.global_variables_initializer())
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    output = sess.run(test_op, feed_dict={a:a_np, b:b_np},
        options=options, run_metadata=run_metadata)
    print(output.shape)
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)

    with open('timeline.ctf.json', 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format())
