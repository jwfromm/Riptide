# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

#%%
import os
import cv2
import tensorflow as tf
import numpy as np
from functools import partial
from riptide.anneal.models.squeezenet import SqueezeNet
from riptide.anneal.models.get_models import get_model
from riptide.anneal.anneal_config import Config


#%%
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


#%%
cifar = tf.keras.datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = cifar

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_images)


#%%
config = Config(quantize=False, a_bits=1, w_bits=1, fixed=False)
with config:
    model = SqueezeNet(classes=10)

model.compile(
    optimizer='adam',
    loss= 'categorical_crossentropy',
    metrics=['accuracy'])

test_input = tf.keras.layers.Input(shape=[32, 32, 3])
test_output = model(test_input)

model.summary()


#%%
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32), steps_per_epoch=len(train_images)/32, epochs=10, validation_data=(test_images, test_labels))


#%%



