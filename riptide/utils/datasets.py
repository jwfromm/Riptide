import tensorflow as tf
import os
from functools import partial

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('record_path', '/data3/imagenet/tfrecords',
        'Path to tfrecord shards.')

def _get_shard_dataset(record_path, split='train'):
    pattern = os.path.join(record_path, split + "*")
    files = tf.data.Dataset.list_files(pattern)
    return files

def _decode_jpeg(image_buffer, size, scope=None):
    with tf.name_scope(values=[image_buffer], name=scope, default_name='decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.resize_images(image, (size, size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def _decode_imagenet(proto, size, preprocess):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    parsed_features = tf.parse_single_example(proto, feature_map)
    features = _decode_jpeg(parsed_features['image/encoded'], size)
    labels = parsed_features['image/class/label']

    if preprocess != None:
        features = preprocess(features)

    return features, tf.cast(labels, tf.int32)

def get_imagenet_dataset(batch_size, split='train', preprocess=None, size=224):
    shard_ds = _get_shard_dataset(FLAGS.record_path, split=split)
    imagenet_ds = shard_ds.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4, sloppy=True))
    decode_fn = partial(_decode_imagenet, size=size, preprocess=preprocess)
    imagenet_ds = imagenet_ds.apply(tf.contrib.data.map_and_batch(
        map_func=decode_fn, batch_size=batch_size, num_parallel_batches=4))
    if split == 'train':
        imagenet_ds = imagenet_ds.shuffle(buffer_size=128)
    imagenet_ds = imagenet_ds.prefetch(batch_size)
    return imagenet_ds
