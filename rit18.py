import scipy.io
from sklearn.feature_extraction.image import extract_patches_2d
from unet import u_net 
import numpy as np
import tensorflow as tf

mat = scipy.io.loadmat("../datasets/rit18_data.mat")


# Extracting training data from the mat dictionnary-----------------------------
print("Extracting training data from the mat dictionnary")

train_data = mat["train_data"].transpose(1,2,0)
train_labels = mat["train_labels"]

val_data = mat["val_data"].transpose(1,2,0)
val_labels = mat["val_labels"]

del mat

# Generating patches from the big image------------------------------------------
print("Generating patches from the big image")
train_data_patches = extract_patches_2d(train_data, (128, 128), max_patches = 0.00005, random_state = 1)
train_labels_patches = extract_patches_2d(train_labels, (128, 128), max_patches=0.00005, random_state = 1)

val_data_patches = extract_patches_2d(val_data, (128, 128), max_patches = 0.00005, random_state = 1)
val_labels_patches = extract_patches_2d(val_labels, (128, 128), max_patches=0.00005, random_state = 1)

del train_data
del train_labels
del val_data
del val_labels

# Removing dark patches taken from the borders of the dataset

indexes = []
for i in range(len(train_data_patches)):
    print(i)
    if np.count_nonzero(train_data_patches[i])/(len(train_data_patches[i].flatten())) != 1.0: 
        indexes.append(i)
train_data_patches = np.delete(train_data_patches, indexes, axis = 0)
train_labels_patches = np.delete(train_labels_patches, indexes, axis = 0)

indexes = []
for i in range(len(val_data_patches)):
    if np.count_nonzero(val_data_patches[i])/(len(val_data_patches[i].flatten())) != 1.0: 
        indexes.append(i)
val_data_patches = np.delete(val_data_patches, indexes, axis = 0)
val_labels_patches = np.delete(val_labels_patches, indexes, axis = 0)

del indexes

train_labels_patches = tf.one_hot(indices = train_labels_patches, depth = 19, dtype=tf.int8)
val_labels_patches = tf.one_hot(indices = val_labels_patches, depth = 19, dtype=tf.int8)

print("Number of patches :", train_data_patches.shape, train_labels_patches.shape)


# Defining the Jaccard-Loss------------------------------------------------------
print("Defining the Jaccard-Loss")

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras import backend as K

def jaccard2_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
def jaccard2_loss(y_true, y_pred, smooth=1.0):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)

# Creating the model--------------------------------------------------------------
print("Creating the model")

model = u_net((128, 128, 7), output_channels=19, activation="relu")
model.compile(loss = "categorical_crossentropy", optimizer=Adam())

print("Fitting the model")
model.fit(train_data_patches, train_labels_patches, batch_size=64, 
            steps_per_epoch=len(train_data_patches) / 64, epochs=5, validation_data = (val_data_patches, val_labels_patches))

