from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, BayesianOptimization
from salinas_utils import load_data
import numpy as np
import tensorflow as tf

from loguru import logger
from pathlib import Path
import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization, Activation, Conv2DTranspose, Dense, Flatten, Reshape, MaxPool1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# -- Throtling GPU use -- 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":

    train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights = load_data.load()

    train_data = train_data.reshape((train_data.shape[0]*train_data.shape[1]*train_data.shape[2], train_data.shape[3]))
    train_labels = train_labels.reshape((train_labels.shape[0]*train_labels.shape[1]*train_labels.shape[2]))

    val_data = val_data.reshape((val_data.shape[0]*val_data.shape[1]*val_data.shape[2], val_data.shape[3]))
    val_labels = val_labels.reshape((val_labels.shape[0]*val_labels.shape[1]*val_labels.shape[2]))

    test_data = test_data.reshape((test_data.shape[0]*test_data.shape[1]*test_data.shape[2], test_data.shape[3]))
    test_labels = test_labels.reshape((test_labels.shape[0]*test_labels.shape[1]*test_labels.shape[2]))

    train_weights = np.ones((train_labels.shape[0])) 
    for i in range(len(classes)):
        train_weights[train_labels == classes[i]] = weights[i]

    val_weights = np.ones((val_labels.shape[0])) 
    for i in range(len(classes)):
        val_weights[val_labels == classes[i]] = weights[i]

    test_weights = np.ones((test_labels.shape[0]))
    for i in range(len(classes)):
        test_weights[test_labels == classes[i]] = weights[i]


    train_labels = tf.one_hot(train_labels, depth=17, dtype=tf.int8).numpy()
    val_labels = tf.one_hot(val_labels, depth=17, dtype=tf.int8).numpy()
    test_labels = tf.one_hot(test_labels, depth=17, dtype=tf.int8).numpy()

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, data, labels, weights, batch_size=32, shuffle=True):
            'Initialization'
            self.data = data
            self.labels = labels
            self.weights = weights
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.data) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Get data and labels
            data_yield = self.data[indexes]
            labels_yield = self.labels[indexes]
            weights_yield = self.weights[indexes]

            return data_yield, labels_yield, weights_yield

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.data))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    batch_size = 3000 
    steps_per_epoch = 20
    train_generator = DataGenerator(train_data, train_labels, train_weights, batch_size)
    val_generator = DataGenerator(val_data, val_labels, val_weights, batch_size)

    print(val_generator.__getitem__(2)[0].shape)

    class CNNHyperModel(HyperModel):
        def __init__(self, input_shape, num_classes):
            self.input_shape = input_shape
            self.num_classes = num_classes

        def build(self, hp):

            nb_filters_0 = hp.Choice("nb_filters_0", values=[8, 16, 32, 64])
            kernel_size = hp.Choice("kernel_size", values=[3, 5, 7])
            kernel_initializer = "glorot_uniform"
            lr = hp.Float(
                        "learning_rate",
                        min_value=1e-4,
                        max_value=1e-2,
                        sampling="LOG",
                        default=1e-3,
                    )

            nb_dense_neurons_1 = hp.Int("nb_dense_neurons_1", min_value=10, max_value=500, step=10, default=100)
            reg = hp.Float("regularization_value", min_value=1e-4, max_value=1, sampling="LOG", default=1e-2)
            reg_dense = hp.Float("reg_dense", min_value=1e-4, max_value=1, sampling="LOG", default=1e-2)
            dropout = hp.Float("dropout", min_value=0., max_value=0.9, step=0.1, default=0.5)

            input_layer = Input(shape=(train_data.shape[1], 1))
            output_channels = 17

            x = Conv1D(filters=nb_filters_0, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=l2(reg), padding='valid')(input_layer)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            #x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout)(x)

            x = Conv1D(filters=nb_filters_0*2, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=l2(reg), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            #x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout)(x)

            x = Conv1D(filters=nb_filters_0*4, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=l2(reg), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            #x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(dropout)(x)

            x = Flatten()(x)
            x = Dense(nb_dense_neurons_1, kernel_regularizer=l2(reg_dense))(x)

            x = Dropout(dropout)(x)
            

            output_layer = Dense(output_channels, activation="softmax")(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=lr), metrics=["accuracy"])

            return model


    def tuner_evaluation(tuner, train_generator, val_generator):

        # Overview of the task
        tuner.search_space_summary()

        # Performs the hyperparameter tuning
        logger.info("Start hyperparameter tuning")
        search_start = time.time()
        tuner.search(train_generator, epochs= 500, steps_per_epoch=steps_per_epoch, validation_data = val_generator, callbacks=[EarlyStopping(monitor='val_accuracy', patience=20)])
        search_end = time.time()
        elapsed_time = search_end - search_start

        # Show a summary of the search
        tuner.results_summary()

        # Retrieve the best model.
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate the best model.
        loss, accuracy = best_model.evaluate(test_data, test_labels, sample_weight=test_weights)
        return elapsed_time, loss, accuracy


    NUM_CLASSES = 17
    INPUT_SHAPE = (train_data.shape[1], 1)

    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)



    
    tuner = BayesianOptimization(hypermodel, objective='val_accuracy', max_trials=500, num_initial_points=2, seed=2, directory = "Salinas_CNN_1D_random_search",  overwrite=True)

    results = []

    elapsed_time, loss, accuracy = tuner_evaluation(
        tuner, train_generator, val_generator
        )
    logger.info(f"Elapsed time = {elapsed_time:10.4f} s, accuracy = {accuracy}, loss = {loss}")
    results.append([elapsed_time, loss, accuracy])
    logger.info(results)