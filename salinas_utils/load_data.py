"""
This code is specifically designed to open Salinas data, preprocess it, and separate it into training, validation, and test sets.
"""

from sklearn.decomposition import PCA 
from scipy.io import loadmat
import tensorflow as tf
import numpy as np




def load(activate_pca = True, pca_explained_variance = 0.9999):

    """
    This function loads the data, preprocesses it, and creates training, validation and test sets. 
    """

    ## -- Loading data and labels --

    data = loadmat('salinas/Salinas_corrected.mat')['salinas_corrected']
    labels = loadmat('salinas/Salinas_gt.mat')['salinas_gt']


    # If PCA is activated, the dimension of each pixel will be reduced
    # For example, after a PCA with 99.99% of explained variance, the number of bands for each pixel will be 29.
    if activate_pca :
        data_shape = data.shape
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        pca = PCA(n_components=pca_explained_variance, svd_solver='full')
        data = pca.fit_transform(data)
        data = data.reshape((data_shape[0], data_shape[1], data.shape[1]))
        del pca


    # Extracting patches from the image
    ksize_row, ksize_col = 32, 32
    ksizes = [1, ksize_row, ksize_col, 1] 
    strides = [1, ksize_row, ksize_col, 1]
    data = np.array([data])
    image_patches = tf.image.extract_patches(data, ksizes, strides, [1, 1, 1, 1], padding="VALID")
    image_patches = tf.squeeze(image_patches).numpy()
    image_patches = image_patches.reshape(image_patches.shape[0], image_patches.shape[1], ksize_row, ksize_col, data.shape[-1])

    # Extracting patches from the ground truth
    labels = labels.reshape(1, labels.shape[0], labels.shape[1], 1)
    labels = tf.image.extract_patches(labels, ksizes, strides, [1, 1, 1, 1], padding="VALID")
    labels = tf.squeeze(labels).numpy()
    labels = labels.reshape(labels.shape[0], labels.shape[1], ksize_row, ksize_col)

    coords = [(i, j) for i in range (0,16) for j in range(0,6)]


   
    # Creating the train, validation and test datasets
    
    coords_val = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (9, 1), (10, 1), (13, 0), (4, 0)]
    coords_test = [(4, 3), (1, 3), (1, 4), (5, 4), (10, 0), (14, 0), (14, 1), (0, 0), (8, 1)]
    dark_coords = [(8, 4), (8, 5), (9, 4), (9, 5), (10, 4), (10, 5), (11, 3), (11, 4), (11, 5), (12, 3), (12, 4), (12, 5), (13,4), (13, 5), (14, 4), (14, 5), (15, 3), (15, 4), (15, 5)]
    coords_train = list(set(coords).difference(set(coords_val)).difference(set(coords_test)).difference(set(dark_coords)))

    train_data = np.array([image_patches[coords_train[i][0], coords_train[i][1]] for i in range(len(coords_train))])
    val_data = np.array([image_patches[coords_val[i][0], coords_val[i][1]] for i in range(len(coords_val))])
    test_data = np.array([image_patches[coords_test[i][0], coords_test[i][1]] for i in range(len(coords_test))])

    train_labels = np.array([labels[coords_train[i][0], coords_train[i][1]] for i in range(len(coords_train))])
    val_labels = np.array([labels[coords_val[i][0], coords_val[i][1]] for i in range(len(coords_val))])
    test_labels = np.array([labels[coords_test[i][0], coords_test[i][1]] for i in range(len(coords_test))])

    # Calculating weights of classes and setting weight of class 0 to zero.
    
    classes, counts = np.unique(labels, return_counts=True)
    counts[0] = 1
    weights = np.sum(counts)/counts
    weights[0] = 0

    return train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights



    

