import os
import tensorflow as tf
from functools import partial

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('f', '', 'kernel')
tf.flags.DEFINE_string('record_path', '/data3/imagenet/tfrecords',
                       'Path to tfrecord shards.')


def _get_shard_dataset(record_path, split='train'):
    pattern = os.path.join(record_path, split + "*")
    files = tf.data.Dataset.list_files(pattern)
    return files


def _decode_jpeg(image_buffer, size, scope=None):
    with tf.name_scope(
            values=[image_buffer], name=scope, default_name='decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.resize_images(image, (size, size))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def _decode_imagenet(proto, preprocess):
    feature_map = {
        'image/encoded':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label':
        tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/class/text':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    parsed_features = tf.parse_single_example(proto, feature_map)
    #features = _decode_jpeg(parsed_features['image/encoded'], size)
    features = tf.image.decode_jpeg(
        parsed_features['image/encoded'], channels=3)
    labels = parsed_features['image/class/label']

    if preprocess != None:
        features = preprocess(features)

    return features, tf.cast(labels, tf.int32)


def get_imagenet_dataset(batch_size, is_training=True, preprocess=None):
    if is_training:
        split = 'train'
    else:
        split = 'val'
    shard_ds = _get_shard_dataset(FLAGS.record_path, split=split)
    imagenet_ds = shard_ds.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=4, sloppy=True))

    decode_fn = partial(_decode_imagenet, preprocess=preprocess)
    imagenet_ds = imagenet_ds.apply(
        tf.contrib.data.map_and_batch(
            map_func=decode_fn, batch_size=batch_size, num_parallel_batches=4))
    if is_training:
        imagenet_ds = imagenet_ds.shuffle(buffer_size=10 * batch_size)
    imagenet_ds = imagenet_ds.prefetch(buffer_size=None)
    return imagenet_ds


def _parse_imagefolder_samples(filename, label, preprocess=None):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    if preprocess is not None:
        image_decoded = preprocess(image_decoded)
    return image_decoded, label


def imagefolder_dataset(root,
                        batch_size,
                        is_training=True,
                        preprocess=None,
                        num_workers=4):
    valid_extensions = ('.png', '.jpg', '.jpeg')
    if is_training:
        split = 'train'
    else:
        split = 'val'
    split_dir = os.path.join(root, split)
    labels_list = os.listdir(split_dir)
    # Iterate through folders and compose list of files and their label.
    images = []
    labels = []
    for i, label in enumerate(labels_list):
        files = os.listdir(os.path.join(split_dir, label))
        for f in files:
            # Make sure found files are valid images.
            if f.lower().endswith(valid_extensions):
                images.append((os.path.join(split_dir, label, f)))
                labels.append(i)
    # Now that dataset is populated, parse it into proper tf dataset.
    decode_fn = partial(_parse_imagefolder_samples, preprocess=preprocess)
    imagenet_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    imagenet_ds = imagenet_ds.apply(
        tf.contrib.data.map_and_batch(
            map_func=decode_fn,
            batch_size=batch_size,
            num_parallel_batches=num_workers))
    if is_training:
        imagenet_ds = imagenet_ds.shuffle(buffer_size=10 * batch_size)
        imagenet_ds = imagenet_ds.prefetch(buffer_size=None)
    return imagenet_ds
