import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from unet import Unet, Unet4, UnetSeparable, cnn_1D
import tensorflow as tf
from landsat_utils.utils import limit_gpu
from scipy.signal import medfilt
import tensorflow as tf


def u_net_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights):

    """This function is the main pipeline for the U-Net.
    
    Parameters :
    ------------
    xxx_data : np.array
    Data of the train, validation and test datasets

    xxx_labels : np.array
    Labels of the train, validation and test datasets

    classes : np.array
    Classes of the dataset

    weights : np.array
    Weights applied for each class during training

    Returns :
    -----------
    Accuracy : float
    Precision : float 
    Recall : float
    F1-Score : float
    Metrics evaluated on the test set
    """

    # Determining the weights of each pixel of the images
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
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    seed = 1
    image_datagen.fit(train_data, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        train_data,
        batch_size=100,
        seed=seed)
    mask_generator = mask_datagen.flow(
        train_labels,
        batch_size=100,
        sample_weight = train_weights,
        seed=seed)

    del train_data, train_labels, train_weights
    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask_squeezed = np.squeeze(mask)
            mask_one_hot = tf.one_hot(mask_squeezed, depth=6, dtype=tf.int8)
            img = tf.convert_to_tensor(img)
            yield img, mask_one_hot

    generator = image_mask_generator(image_generator, mask_generator)

    # One-hot encoding the validation labels
    val_labels = tf.one_hot(val_labels, depth=6, dtype=tf.int8)

    ## -- Training the Unet Model --

    # Image shape

    img_rows = 256
    img_cols = 256
    img_channels = val_data.shape[3]

    #Output
    nb_classes = 6

    # Architecture Parameters
    nb_filters_0 = 16

    # Saving the model at each step
    checkpoint_filepath = "tmp/checkpoint_SPARCS_unet"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Early Stopping Callback
    ES = EarlyStopping(monitor='val_accuracy', patience=20)

    # Deep Learning Model

    model = Unet4((img_rows, img_cols, img_channels), nb_filters=nb_filters_0, output_channels=nb_classes, initialization="he_normal", kernel_size=5, drop=0.0, regularization=l2(0.000135))
    print(model.summary())
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=0.00019), metrics=["accuracy"])

    history = model.fit(generator, batch_size=100, 
                steps_per_epoch=10, 
                epochs=500,
                validation_data = (val_data, val_labels, val_weights),
                callbacks=[model_checkpoint_callback, ES])
    
    # Making a prediction on the test data
    pred = model.predict(test_data)

    # Computing metrics
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(np.argmax(pred, axis=3), test_labels)

    recall = tf.keras.metrics.Recall()
    recall.update_state(tf.one_hot(test_labels, depth=6).numpy().flatten(), pred.flatten())

    precision = tf.keras.metrics.Precision()
    precision.update_state(tf.one_hot(test_labels, depth=6).numpy().flatten(), pred.flatten())

    # Computing the confusion matrix
    print(confusion_matrix(np.argmax(pred, axis=3).flatten(), test_labels.flatten(), sample_weight=test_weights.flatten()))

    # Plotting images of the test dataset
    c = 0
    cmap = plt.get_cmap('viridis', 6)
    for image in pred[:50]:
        plt.imshow(np.argmax(image, axis=2)+1e-5, cmap=cmap, vmin=0, vmax=6)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/u_net/pred{c}")
        plt.clf()

        plt.imshow(test_labels[c]+1e-5, vmin=0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/u_net/GT{c}")
        plt.clf()

        plt.imshow(test_data[c][:,:,5:9]/np.amax(test_data[c][:,:,5:9]))
        plt.savefig(f"images/SPARCS/u_net/original{c}")

        c+=1
    
    pred = np.argmax(pred, axis=3)
    pred = pred.flatten()
    test_labels = test_labels.flatten()

    # Computing the confusion matrix
    conf = confusion_matrix(test_labels, pred, labels=[0, 1, 2, 3, 4, 5])
    np.save("logs/metrics/SPARCS/u_net/confusion_matrix", conf)

    # Computing metrics
    test_accuracy = accuracy.result().numpy()
    test_recall = recall.result().numpy()
    test_precision = precision.result().numpy()
    test_f1 = 2/(1/test_recall + 1/test_precision)
    
    print("Test accuracy = ", test_accuracy)
    print("Test recall =", test_recall)
    print("Test precision=", test_precision)
    print("Test f1 =", test_f1)


def u_net_sep_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights):

     """This function is the main pipeline for the separable U-Net.
    
    Parameters :
    ------------
    xxx_data : np.array
    Data of the train, validation and test datasets

    xxx_labels : np.array
    Labels of the train, validation and test datasets

    classes : np.array
    Classes of the dataset

    weights : np.array
    Weights applied for each class during training

    Returns :
    -----------
    Accuracy : float
    Precision : float 
    Recall : float
    F1-Score : float
    Metrics evaluated on the test set
    """
    
    l, x, y = train_labels.shape
    new_shape = (l, x, y, 1)
    train_labels = np.reshape(train_labels, new_shape)

    # Calculating weights of each pixel
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
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    print(train_data.shape)
    seed = 1
    image_datagen.fit(train_data, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        train_data,
        batch_size=20,
        seed=seed)
    mask_generator = mask_datagen.flow(
        train_labels,
        batch_size=20,
        seed=seed)
    
    del train_data, train_labels, train_weights

    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask_squeezed = np.squeeze(mask)
            mask_one_hot = tf.one_hot(mask_squeezed, depth=6, dtype=tf.int8)
            img = tf.convert_to_tensor(img)
            yield img, mask_one_hot

    generator = image_mask_generator(image_generator, mask_generator)

    # One-hot encoding the validation labels
    val_labels = tf.one_hot(val_labels, depth=6, dtype=tf.int8)


    ## -- Training the Unet Model --

    # Image shape

    img_rows = 256
    img_cols = 256
    img_channels = val_data.shape[3]

    #Output
    nb_classes = 6

    # Saving the model at each step
    checkpoint_filepath = "tmp/checkpoint_SPARCS_unet"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Early Stopping Callback
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)

    # Deep Learning Model
    
    model = UnetSeparable(shape=(256, 256, 10), depth=10, nb_filters=4, kernel_size=5, initialization="he_normal", output_channels=6, drop=0.3, regularization=l2(0.00013443))
    print(model.summary())

    model.compile(loss = "categorical_crossentropy", optimizer=Adam(0.00046131), metrics=["accuracy"])

    # Training the model
    history = model.fit(generator, batch_size=20, 
                steps_per_epoch=50, 
                epochs=150,
                validation_data = (val_data, val_labels),
                callbacks=[model_checkpoint_callback])
    
    np.save("logs/metrics/SPARCS/u_net_sep/train_accuracy", history.history['accuracy'])
    np.save("logs/metrics/SPARCS/u_net_sep/val_accuracy", history.history['val_accuracy'])

    # Making a prediction on the test data
    pred = model.predict(test_data)

    # Computing metrics on the test data
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(np.argmax(pred, axis=3), test_labels)

    recall = tf.keras.metrics.Recall()
    recall.update_state(tf.one_hot(test_labels, depth=6).numpy().flatten(), pred.flatten())

    precision = tf.keras.metrics.Precision()
    precision.update_state(tf.one_hot(test_labels, depth=6).numpy().flatten(), pred.flatten())

    # Plotting result images on the test dataset
    import matplotlib.pyplot as plt
    c = 0
    cmap = plt.get_cmap('viridis', 6)
    for image in pred:
        plt.imshow(np.argmax(image, axis=2), vmin= 0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/u_net_sep/pred{c}")
        plt.clf()

        plt.imshow(test_labels[c], vmin=0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/u_net_sep/GT{c}")
        plt.clf()

        plt.imshow(test_data[c][:,:,5:9]/np.amax(test_data[c][:,:,5:9]))
        plt.savefig(f"images/SPARCS/u_net_sep/original{c}")

        c+=1
 
    pred = np.argmax(pred, axis=3)
    pred = pred.flatten()
    test_labels = test_labels.flatten()

    # Computing the confusion matrix
    conf = confusion_matrix(test_labels, pred)
    print(conf)
    np.save("logs/metrics/SPARCS/u_net_sep/confusion_matrix", conf)

    test_accuracy = accuracy.result().numpy()
    test_recall = recall.result().numpy()
    test_precision = precision.result().numpy()
    test_f1 = 2/(1/test_recall + 1/test_precision)
    
    print("Test accuracy = ", test_accuracy)
    print("Test recall =", test_recall)
    print("Test precision=", test_precision)
    print("Test f1 =", test_f1)
    


def cnn_1d_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights):

    """This function is the main pipeline for the 1D CNN.
    
    Parameters :
    ------------
    xxx_data : np.array
    Data of the train, validation and test datasets

    xxx_labels : np.array
    Labels of the train, validation and test datasets

    classes : np.array
    Classes of the dataset

    weights : np.array
    Weights applied for each class during training

    Returns :
    -----------
    Accuracy : float
    Precision : float 
    Recall : float
    F1-Score : float
    Metrics evaluated on the test set
    """

    # Reshaping the data

    train_data = train_data.reshape(train_data.shape[0]*train_data.shape[1]*train_data.shape[2], train_data.shape[3])
    train_labels = train_labels.reshape(train_labels.shape[0]*train_labels.shape[1]*train_labels.shape[2])

    val_data = val_data.reshape(val_data.shape[0]*val_data.shape[1]*val_data.shape[2], val_data.shape[3])
    val_labels = val_labels.reshape(val_labels.shape[0]*val_labels.shape[1]*val_labels.shape[2])
    
    test_shape = test_labels.shape
    test_data = test_data.reshape(test_data.shape[0]*test_data.shape[1]*test_data.shape[2], test_data.shape[3])
    test_labels = test_labels.reshape(test_labels.shape[0]*test_labels.shape[1]*test_labels.shape[2])


    # Computing the weights of each pixel
    train_weights = np.ones((train_labels.shape[0])) 
    for i in range(len(classes)):
        train_weights[train_labels == classes[i]] = weights[i]

    val_weights = np.ones((val_labels.shape[0])) 
    for i in range(len(classes)):
        val_weights[val_labels == classes[i]] = weights[i]

    test_weights = np.ones((test_labels.shape[0])) 
    for i in range(len(classes)):
        test_weights[test_labels == classes[i]] = weights[i]

     # One-Hot Encoding the labels
    train_labels = tf.one_hot(train_labels, depth=6, dtype=tf.int8).numpy()
    val_labels = tf.one_hot(val_labels, depth=6, dtype=tf.int8).numpy()



    # Creating a data generator class

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, data, labels, weights, batch_size=32, n_classes=10, shuffle=True):
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
    
    
    ## -- Training the model --

    batch_size = 30000 
    steps_per_epoch = 100
    train_generator = DataGenerator(train_data, train_labels, train_weights, batch_size, shuffle=True)
    val_generator = DataGenerator(val_data, val_labels, val_weights, batch_size, shuffle=True)

    checkpoint_filepath = "tmp/checkpoint_SPARCS_cnn_1D"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model = cnn_1D(shape=(10, 1), kernel_size=3, nb_filters_0=128, nb_dense_neurons=180, kernel_reg=0.0083, dense_reg=0.054, output_channels=6, dropout=0.0, learning_rate=0.000322)
    print(model.summary())

    # Training the model
    history = model.fit(train_generator,
                steps_per_epoch=steps_per_epoch, 
                epochs=500,
                validation_data = val_generator,
                validation_steps = 5,
                callbacks=[EarlyStopping(monitor='val_accuracy', patience=40)]
                )


    # Making a prediction
    y_pred = model.predict(test_data, batch_size=30000)
    y_pred = np.argmax(y_pred, axis=1)

    # Computing accuracy
    accuracy = Accuracy()
    accuracy.update_state(test_labels, y_pred, sample_weight=test_weights)
    pred_accuracy = accuracy.result()
    print(pred_accuracy)

    # Computing the confusion matrix
    print("Computing confusion matrix")
    conf = confusion_matrix(test_labels, y_pred)
    print(conf)
    np.save("logs/metrics/SPARCS/cnn_1D/confusion_matrix", conf)


    y_pred = y_pred.reshape(test_shape)
    print(y_pred.shape)

    # Plotting result images on the test dataset
    c = 0
    cmap = plt.get_cmap('viridis', 6)
    for image in y_pred[:30]:
        plt.imshow(image+1e-5, vmin=0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/cnn_1D/pred{c}_raw")
        plt.clf()

        plt.imshow(test_labels.reshape(test_shape)[c]+1e-5, vmin=0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/cnn_1D/GT{c}")
        plt.clf()

        c+=1

    # Applying a median filter on the results
    for i in range(len(y_pred)):
        y_pred[i] = medfilt(y_pred[i], kernel_size = 3)



    # Plotting the median filtered result images
    c = 0
    for image in y_pred[:30]:
        plt.imshow(image+1e-5, vmin=0, vmax=6, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/SPARCS/cnn_1D/pred{c}_filtered")
        plt.clf()

        c+=1
    
    # Computing metrics on the test dataset
    y_pred = y_pred.reshape(test_shape[0]*test_shape[1]*test_shape[2])
    accuracy = Accuracy()
    accuracy.update_state(test_labels, y_pred, sample_weight=test_weights)


    test_labels = tf.one_hot(test_labels, depth=6).numpy().flatten()
    y_pred = tf.one_hot(y_pred, depth=6).numpy().flatten()
    recall = tf.keras.metrics.Recall()
    recall.update_state(test_labels, y_pred)

    precision = tf.keras.metrics.Precision()
    precision.update_state(test_labels, y_pred)


    # Printing metrics
    test_accuracy = accuracy.result().numpy()
    test_recall = recall.result().numpy()
    test_precision = precision.result().numpy()
    test_f1 = 2/(1/test_recall + 1/test_precision)
    
    print("Test accuracy = ", test_accuracy)
    print("Test recall =", test_recall)
    print("Test precision=", test_precision)
    print("Test f1 =", test_f1)


