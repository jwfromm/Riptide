import tensorflow as tf
import os

def eval_image(image, height=224, width=224, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(values=[image, height, width], name=scope,
                     default_name='eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return tf.cast(image, dtype=tf.float32)

def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def tfrecord_dataset(path, prefix, preprocess=None, shuffle=False, repeat_count=1, batch_size=1, workers=4, return_dataset=True):
    pattern = os.path.join(path, prefix + "-*")
    files = tf.data.Dataset.list_files(pattern)

    #with tf.device('/cpu:0'):
    def _decode(proto):
        feature_map = {
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
          'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
          'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value=''),
        }
        parsed_features = tf.parse_single_example(proto, feature_map)
        features = decode_jpeg(parsed_features["image/encoded"])
        labels = parsed_features["image/class/label"]
        if preprocess is None:
            features = eval_image(features)
        else:
            features = preprocess(features)
        return features, tf.cast(labels, tf.int32)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=workers, sloppy=True))
    #dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=32, block_length=batch_size)
    #dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_decode, batch_size=batch_size, num_parallel_batches=workers))
    if shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=128)
    if repeat_count is None:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times        
    #dataset = dataset.map(_decode)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(4*batch_size) # prefetch samples
    if return_dataset:
        return dataset
    else:
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        batch_labels = tf.squeeze(tf.one_hot(batch_labels, 1000), axis=1)
        return {'input_1': batch_features}, batch_labels
