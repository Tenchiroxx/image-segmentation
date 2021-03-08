import os
import tifffile as tif
import numpy as np
from unet import unet4, u_net
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight



# -- Loading the datasets --

filenames = [file[:-8] for file in os.listdir("./data-landsat/") if "_data.tif" in file]

data = np.array([tif.imread("data-landsat/" + filenames[i]+"data.tif") for i in range(len(filenames))])
labels = np.array([tif.imread("data-landsat/" + filenames[i]+"labels.tif") for i in range(len(filenames))])

print("Initial length :", data.shape)

#Computing class imbalance
unique, counts = np.unique(labels.flatten(), return_counts=True)
class_weights = dict(zip(unique, labels.size/counts))


# -- Data Preprocessing --

#Splitting all images in four patches

n_split = 4
size = 1024//n_split
stride = (1000-size)//(n_split-1)

data_temp = np.zeros((n_split**2*len(data), size, size, 10))
label_temp = np.zeros((n_split**2*len(data), size, size))

for i in range(n_split):
  for j in range(n_split):
    data_temp[(i*n_split+j)*len(data):(i*n_split+j+1)*len(data)] = data[:, i*stride:i*stride+size, j*stride:j*stride+size, :]
    label_temp[(i*n_split+j)*len(data):(i*n_split+j+1)*len(data)] = labels[:, i*stride:i*stride+size, j*stride:j*stride+size]

data = data_temp
labels = label_temp

del data_temp

## Splitting into training, validation and testing dataset
train_data, train_labels = data[:n_split**2*61], labels[:n_split**2*61]
val_data, val_labels = data[n_split**2*61:n_split**2*71], labels[n_split**2*61:n_split**2*71]
test_data, test_labels = data[n_split**2*71:], labels[n_split**2*71:]

print(train_data.shape, val_data.shape, test_data.shape)
del data
del labels

## One-Hot Encoding & Image Augmentation using an Image Generator

image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

seed = 1

l, x, y = train_labels.shape
new_shape = (l, x, y, 1)
train_labels = np.reshape(train_labels, new_shape)

image_datagen.fit(train_data, augment=True, seed=seed)
mask_datagen.fit(train_labels, augment=True, seed=seed)

# Giving more weight to rare classes
sample_weights = np.ones(shape=train_labels.shape)
for i in range(0, 7):
  	sample_weights[train_labels == i] = class_weights[i]

image_generator = image_datagen.flow(
    train_data,
    seed=seed)
mask_generator = mask_datagen.flow(
    train_labels,
    sample_weight = sample_weights,
    seed=seed)

del train_data
del train_labels

def image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        mask_squeezed = np.squeeze(mask)
        mask_one_hot = tf.one_hot(mask_squeezed, depth=7, dtype=tf.int8)
        img = tf.convert_to_tensor(img)
        yield img, mask_one_hot

generator = image_mask_generator(image_generator, mask_generator)

val_labels = tf.one_hot(val_labels, depth=7, dtype=tf.int8)
test_labels = tf.one_hot(test_labels, depth=7, dtype=tf.int8)


## -- Training the Unet Model --

# Image shape
img_rows = 256
img_cols = 256
img_channels = 10

#Output
nb_classes = 7 

# Architecture Parameters
nb_filters_0 = 32

# Saving the model at each step
checkpoint_filepath = "tmp/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


#Deep Learning Model
# model = unet4(img_rows, img_cols, img_channels, nclasses=nb_classes, filters=nb_filters_0)
# model.compile(loss = "categorical_crossentropy", optimizer=Adam())
# model.fit(generator, batch_size=32, 
#             steps_per_epoch=30, 
#             epochs=200,
#             validation_data = (val_data, val_labels),

#             callbacks=[model_checkpoint_callback])


# Deep Learning Model 2
model = u_net((img_rows, img_cols, img_channels), output_channels=7, initialization="he_normal")
model.compile(loss = "categorical_crossentropy", optimizer=Adam())
model.fit(generator, batch_size=32, 
            steps_per_epoch=30, 
            epochs=5,
            validation_data = (val_data, val_labels),

            callbacks=[model_checkpoint_callback])

