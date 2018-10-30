import os
import random
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('root', '/data3/imagenet/',
                       'Path to dataset to convert to tfrecords.')
tf.flags.DEFINE_string('record_path', '/data3/imagenet/tfrecords',
                       'Output path of new tfrecords.')
tf.flags.DEFINE_string(
    'split', 'train',
    'Which split to load. This affects both the output record names'
    'and which images are loaded from root.')

tf.flags.DEFINE_integer('image_size', None,
                        'Height and width to resize record examples to.')

tf.flags.DEFINE_integer('num_shards', 8, 'Number of record shards to create.')

tf.flags.DEFINE_integer(
    'num_workers', 8,
    'Number of threads to work on creating the record dataset.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Might be useful for applying some preprocessing on the images.
# Currently unused.
def load_image(sess, path, image_size):
    image_string = tf.read_file(tf.convert_to_tensor(path))
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    if image_size is not None:
        image_decoded = tf.image.resize_images(image_decoded,
                                               [image_size, image_size])
    image_encoded = tf.image.encode_jpeg(image_decoded)
    image_np = sess.run(image_encoded)
    return image_np


def load_dataset(root, shuffle):
    valid_extensions = ('.png', '.jpg', '.jpeg')
    labels_list = os.listdir(root)
    labels_list.sort()
    # Iterate through folders and compose list of files and label
    samples = []
    for i, label in enumerate(labels_list):
        files = os.listdir(os.path.join(root, label))
        for f in files:
            # Confirm found files are valid images.
            if f.lower().endswith(valid_extensions):
                samples.append((os.path.join(root, label, f), i, label))
    # Shuffle if specified.
    if shuffle:
        random.shuffle(samples)
    return samples


def create_example(sess, sample):
    filename, label, label_name = sample
    # Directly read encoded jpeg bytes.
    image_raw = open(filename, 'rb').read()
    #image_raw = load_image(sess, filename, FLAGS.image_size)
    #image_raw = image.tostring()
    #label_name_raw = label_name.tostring()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': _bytes_feature(image_raw),
                'label': _int64_feature(int(label))
                #'label_name': _bytes_feature(label_name_raw)
            }))
    return example


def worker_fn(shard, samples):
    # Create writer for this worker.
    sess = tf.Session()
    filename = os.path.join(FLAGS.record_path, "%s_%d.tfrecord" % (FLAGS.split,
                                                                   shard))
    with tf.python_io.TFRecordWriter(filename) as writer:
        for i, s in enumerate(samples):
            example = create_example(sess, s)
            writer.write(example.SerializeToString())
            if i % 500 == 0:
                print("Shard %d: Finished sample %d." % (shard, i))


if __name__ == '__main__':
    # Turn off GPUS
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    shuffle = (FLAGS.split == 'train')
    dataset = load_dataset(os.path.join(FLAGS.root, FLAGS.split), shuffle)
    dataset_chunks = np.array_split(dataset, FLAGS.num_shards)
    worker_args = zip(range(FLAGS.num_shards), dataset_chunks)
    # Start up a thread pool and launch worker jobs.
    p = Pool(FLAGS.num_workers)
    p.starmap(worker_fn, worker_args)
