import tensorflow as tf
import numpy as np
from functools import partial
from datasets import imagerecord_dataset, imagefolder_dataset
from riptide.utils.preprocessing.inception_preprocessing import preprocess_image


class DatasetTest(tf.test.TestCase):
    def test_get_dataset(self):
        #preprocess = tf.keras.applications.inception_v3.preprocess_input
        preprocess = partial(
            preprocess_image, height=224, width=224, is_training=True)
        ds = imagerecord_dataset(
            '/data/imagenet/tfrecords', 2, preprocess=preprocess)
        image, label = next(iter(ds))
        self.assertEqual((2, 224, 224, 3), image.shape)
        self.assertTrue((np.abs(image) <= 1.0).all())

    def test_imagefolder(self):
        preprocess = partial(
            preprocess_image, height=224, width=224, is_training=True)
        ds = imagefolder_dataset(
            root='/data/imagenet', batch_size=2, preprocess=preprocess)
        image, label = next(iter(ds))
        self.assertEqual((2, 224, 224, 3), image.shape)
        self.assertTrue((np.abs(image) <= 1.0).all())


if __name__ == '__main__':
    tf.test.main()
