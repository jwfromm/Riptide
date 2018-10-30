import tensorflow as tf
import numpy as np
from functools import partial
from datasets import imagerecord_dataset, imagefolder_dataset
from models.research.slim.preprocessing.inception_preprocessing import preprocess_image


class DatasetTest(tf.test.TestCase):
    def test_get_dataset(self):
        #preprocess = tf.keras.applications.inception_v3.preprocess_input
        preprocess = partial(
            preprocess_image, height=224, width=224, is_training=True)
        ds = imagerecord_dataset(2, preprocess=preprocess)
        image, label = ds.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            image_out = sess.run(image)
            self.assertEqual((2, 224, 224, 3), image_out.shape)
            self.assertTrue((np.abs(image_out) <= 1.0).all())

    def test_imagefolder(self):
        preprocess = partial(
            preprocess_image, height=224, width=224, is_training=True)
        ds = imagefolder_dataset(
            root='/data2/imagenet', batch_size=2, preprocess=preprocess)
        image, label = ds.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            image_out = sess.run(image)
            self.assertEqual((2, 224, 224, 3), image_out.shape)
            self.assertTrue((np.abs(image_out) <= 1.0).all())


if __name__ == '__main__':
    tf.test.main()
