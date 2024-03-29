from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, BayesianOptimization
from landsat_utils import load_data
import numpy as np
import tensorflow as tf
from landsat_utils.utils import limit_gpu
from loguru import logger
from pathlib import Path
import os
import tifffile as tif
from scipy.io import loadmat
import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization, Activation, Conv2DTranspose, Dense, Flatten, Reshape, MaxPool1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""This tuner is used to find the best hyperparameters for the U-Net model when reproducing Jeppesen's experiment with all bands"""

# -- Throtling GPU use -- 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":


    filenames = [file[:-8] for file in os.listdir("./data-landsat/") if "_data.tif" in file]

    data = np.array([tif.imread("data-landsat/" + filenames[i]+"data.tif") for i in range(len(filenames))])
    labels = np.array([tif.imread("data-landsat/" + filenames[i]+"labels.tif") for i in range(len(filenames))])


    print(data.shape)


    simplifier = {0:0, 1:0, 2:2, 3:2, 4:2, 5:1, 6:2}
    for key in simplifier.keys():
        labels[labels == key] = simplifier[key]

    #Computing class imbalance

    classes, counts = np.unique(labels.flatten(), return_counts=True)
    weights = sum(counts)/counts

    #Extracting patches

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

    np.random.seed(seed=2)
    indexes = np.arange(len(image_patches)) 
    np.random.shuffle(indexes)

    train_indexes = indexes[:int(0.9*len(image_patches))]
    val_indexes = indexes[int(0.9*len(image_patches)):int(0.95*len(image_patches))]
    test_indexes = indexes[int(0.95*len(image_patches)):] 



    train_data, train_labels = image_patches[train_indexes], labels[train_indexes]
    val_data, val_labels = image_patches[val_indexes], labels[val_indexes]
    test_data, test_labels = image_patches[test_indexes], labels[test_indexes] 


    l, x, y = train_labels.shape
    new_shape = (l, x, y, 1)
    train_labels = np.reshape(train_labels, new_shape)



    train_weights = np.ones(shape=train_labels.shape)
    for i in range(len(classes)):
        train_weights[train_labels == classes[i]] = weights[i]

    val_weights = np.ones(shape=val_labels.shape)
    for i in range(len(classes)):
        val_weights[val_labels == classes[i]] = weights[i]

    test_weights = np.ones(shape=test_labels.shape)
    for i in range(len(classes)):
        test_weights[test_labels == classes[i]] = weights[i]


    # Creating an image generator
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, dtype=np.int16)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    seed = 1
    image_datagen.fit(train_data, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        train_data,
        batch_size=20,
        seed=seed)
    mask_generator = mask_datagen.flow(
        train_labels,
        sample_weight = train_weights,
        batch_size=20,
        seed=seed)

    print(np.unique(train_labels))
    print(train_data.shape, val_data.shape, train_weights.shape)
    del train_data, train_labels, train_weights
    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask_squeezed = np.squeeze(mask)
            mask_one_hot = tf.one_hot(mask_squeezed, depth=3, dtype=tf.int8)
            img = tf.convert_to_tensor(img, dtype=tf.int16)
            yield img, mask_one_hot

    generator = image_mask_generator(image_generator, mask_generator)

    # One-hot encoding the validation labels
    val_labels = tf.one_hot(val_labels, depth=3, dtype=tf.int8)

    ## -- Training the Unet Model --

    # Image shape

    img_rows = 256
    img_cols = 256
    img_channels = val_data.shape[3]
    shape = (img_rows, img_cols, img_channels)
    #Output
    nb_classes = 3


    ES = EarlyStopping(monitor='val_accuracy', patience=20)


    ############################################################################
    ############################################################################

    class CNNHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes

        def build(self, hp):

            def ConvBlock(tensor, nb_filters, kernel_size=3, padding='same', initializer='glorot_uniform', activation='relu'):
                x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=l2(reg))(tensor)
                x = BatchNormalization()(x)
                x = Activation(activation)(x)
                x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=l2(reg))(x)
                x = BatchNormalization()(x)
                x = Activation(activation)(x)
                return x

            def DeconvBlock(tensor, residual, nb_filters, kernel_size=3, padding="same", strides=(2,2)):
                y = Conv2DTranspose(nb_filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding=padding)(tensor)
                y = concatenate([y, residual], axis=3)
                y = ConvBlock(y, nb_filters, kernel_size)
                return y


            

            nb_filters = hp.Choice("nb_filters_0", values=[16, 32, 64])
            drop = hp.Float("dropout", min_value=0., max_value=0.9, step=0.1, default=0.5)
            reg = hp.Float("regularization_value", min_value=1e-4, max_value=1, sampling="LOG", default=1e-2)
            lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG", default=1e-3)
            kernel_size = hp.Choice("kernel_size", values=[3, 5])

            initialization="glorot_uniform"
            activation="relu"
            exp=1
            sigma_noise=0
            output_channels=3

            input_layer = Input(shape=shape)

            conv1 = ConvBlock(input_layer, nb_filters=nb_filters, kernel_size=kernel_size, initializer=initialization, activation=activation)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            if drop > 0.0: pool1 = Dropout(drop)(pool1)

            conv2 = ConvBlock(pool1, nb_filters=nb_filters * 2 **(1 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation )
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            if drop > 0.0: pool2 = Dropout(drop)(pool2)

            conv3 = ConvBlock(pool2, nb_filters=nb_filters * 2 **(2 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation )
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            if drop > 0.0: pool3 = Dropout(drop)(pool3)

            conv4 = ConvBlock(pool3, nb_filters=nb_filters * 2 **(3 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation )
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            if drop > 0.0: pool4 = Dropout(drop)(pool4)

            

            deconv5 = DeconvBlock(conv4, residual=conv3, nb_filters=nb_filters * 2 **(2 * exp), kernel_size=kernel_size)
            if drop > 0.0: deconv5 = Dropout(drop)(deconv5)

            deconv6 = DeconvBlock(deconv5, residual=conv2, nb_filters=nb_filters * 2 **(1 * exp), kernel_size=kernel_size)
            if drop > 0.0: deconv6 = Dropout(drop)(deconv6)

            deconv7 = DeconvBlock(deconv6, residual=conv1, nb_filters=nb_filters, kernel_size=kernel_size)
            if drop > 0.0: deconv7 = Dropout(drop)(deconv7)



            if sigma_noise > 0:
                deconv7 = GaussianNoise(sigma_noise)(deconv7)

            output_layer = Conv2D(filters=output_channels, kernel_size=(1, 1))(deconv7)
            output_layer = BatchNormalization()(output_layer)
            output_layer = Activation('softmax')(output_layer)


            model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
            model.compile(loss = "categorical_crossentropy", optimizer=Adam(lr), metrics=["accuracy"])
            return model



    def tuner_evaluation(tuner, generator, val_data, val_labels, val_weights):

        # Overview of the task
        tuner.search_space_summary()

        # Performs the hyperparameter tuning
        logger.info("Start hyperparameter tuning")
        search_start = time.time()
        tuner.search(generator, epochs= 500, batch_size=20, steps_per_epoch=50, validation_data = (val_data, val_labels, val_weights), callbacks=[EarlyStopping(monitor='accuracy', patience=20)])
        search_end = time.time()
        elapsed_time = search_end - search_start

        # Show a summary of the search
        tuner.results_summary()

        # Retrieve the best model.
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate the best model.
        loss, accuracy = best_model.evaluate(test_data, test_labels, sample_weight=test_weights)
        return elapsed_time, loss, accuracy


    NUM_CLASSES = 3
    INPUT_SHAPE = shape

    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

 

    #tuner = RandomSearch(hypermodel, objective='accuracy', max_trials=500, seed=2, directory = "SPARCS_unet_random_search", max_model_size=100000000, overwrite=True)
    tuner = BayesianOptimization(hypermodel, objective='accuracy', max_trials=500, num_initial_points=5, seed=2, directory = "SPARCS_unet_random_search", max_model_size=100000000, overwrite=True)

    results = []

    elapsed_time, loss, accuracy = tuner_evaluation(
        tuner, generator, val_data, val_labels, val_weights
        )
    logger.info(f"Elapsed time = {elapsed_time:10.4f} s, accuracy = {accuracy}, loss = {loss}")
    results.append([elapsed_time, loss, accuracy])
    logger.info(results)