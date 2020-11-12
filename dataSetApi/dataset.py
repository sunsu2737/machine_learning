import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# print(type(flowers), flowers)
flowers_root = os.path.join(os.path.dirname(flowers), 'flower_photos')
# def gen_it():
#     return img_gen.flow_from_directory(flowers_root)

ds = tf.data.Dataset.from_generator(
    # gen_it,
    lambda: img_gen.flow_from_directory(flowers_root), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([32,256,256,3], [32,5])
)

# print(ds.element_spec)
for images, label in ds.take(1):
  print('images.shape: ', images.shape)
  print('labels.shape: ', label.shape)
# print(flowers_root)
# class_names = os.listdir(flowers_root)
# print(class_names)
# os.remove(os.path.join(flowers_root, 'LICENSE.txt'))
# class_names = os.listdir(flowers_root)
# print(class_names)

# train, test = tf.keras.datasets.fashion_mnist.load_data()


# images, labels = train    # images and labels are numpy arrays
# images = images/255

# print(type(images))
# print(len(labels))

# dataset = tf.data.Dataset.from_tensor_slices((images, labels))
# print(dataset.element_spec)

# for img, label in dataset.take(3):
#   plt.imshow(img)
#   plt.show()
#   print(label.numpy())

# def gen_series():
#   i = 0
#   while True:
#     size = np.random.randint(0, 10)
#     yield i, np.random.normal(size=(size,))
#     i += 1
# ds_series = tf.data.Dataset.from_generator(
#     gen_series,
#     output_types=(tf.int32, tf.float32),
#     output_shapes=((), (None,)))
# dataset = tf.data.Dataset.from_tensor_slices(np.array([8, 3, 0, 8, 2, 1]))
# print(dataset)
# print(dataset.element_spec)

# for elem in dataset:
#   print(elem)
#   print(elem.numpy())

# it = iter(dataset)
# while True:
#   try:
#     print(next(it).numpy())
#   except Exception as e:
#     break

# print(tf.random.uniform([4, 10]))
# dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
# print(dataset1.element_spec)

# for elem in dataset1:
#   print(elem)
#   print(elem.numpy())

# dataset2 = tf.data.Dataset.from_tensor_slices(
#    (tf.random.uniform([4]),
#     tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

# print(dataset2.element_spec)

# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
# print(dataset3.element_spec)
# for elem in dataset3:
#   print(elem)
