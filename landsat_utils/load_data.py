import numpy as np
import tensorflow as tf
import tifffile as tif 
import os
from landsat_utils.utils import limit_gpu

def load():


    """
    This function loads the data, preprocesses it, and creates training, validation and test sets. 
    """

    #Opening the data files
    
    filenames = [file[:-8] for file in os.listdir("./data-landsat/") if "_data.tif" in file]

    data = np.array([tif.imread("data-landsat/" + filenames[i]+"data.tif") for i in range(len(filenames))])
    labels = np.array([tif.imread("data-landsat/" + filenames[i]+"labels.tif") for i in range(len(filenames))])

    labels[labels==6] = 2

    #Computing class imbalance

    classes, counts = np.unique(labels.flatten(), return_counts=True)
    weights = sum(counts)/counts

    #Determining patch sizes

    ksize_row, ksize_col = 256, 256
    ksizes = [1, ksize_row, ksize_col, 1] 
    strides = [1, 248, 248, 1]

    # Extracting patches from the image
    image_patches = tf.image.extract_patches(data, ksizes, strides, [1, 1, 1, 1], padding="VALID")
    image_patches = tf.squeeze(image_patches).numpy()
    image_patches = image_patches.reshape(image_patches.shape[0]*image_patches.shape[1]*image_patches.shape[2], ksize_row, ksize_col, data.shape[-1])


    # Extracting patches from the ground truth
    labels = labels.reshape(labels.shape[0], labels.shape[1], labels.shape[2], 1)
    labels = tf.image.extract_patches(labels, ksizes, strides, [1, 1, 1, 1], padding="VALID")
    labels = tf.squeeze(labels).numpy()
    labels = labels.reshape(labels.shape[0]*labels.shape[1]*labels.shape[2], ksize_row, ksize_col) 


    #Creating train, validation and test datasets
    np.random.seed(seed=2)
    indexes = np.arange(len(image_patches)) 
    np.random.shuffle(indexes)
    train_indexes = indexes[:int(0.9*len(image_patches))]
    val_indexes = indexes[int(0.9*len(image_patches)):int(0.95*len(image_patches))]
    test_indexes = indexes[int(0.95*len(image_patches)):] 



    train_data, train_labels = image_patches[train_indexes], labels[train_indexes]
    val_data, val_labels = image_patches[val_indexes], labels[val_indexes]
    test_data, test_labels = image_patches[test_indexes], labels[test_indexes] 

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights

