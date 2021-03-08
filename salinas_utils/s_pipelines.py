from unet import Unet4, UnetSeparable, cnn_1D
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import Accuracy 
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.regularizers import l2 
from scipy.signal import medfilt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


def u_net_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights):


    # Creating an image generator
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    seed = 1
    l, x, y = train_labels.shape
    new_shape = (l, x, y, 1)
    train_labels = np.reshape(train_labels, new_shape)

    image_datagen.fit(train_data, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    sample_weights = np.ones(shape=train_labels.shape)
    for i in range(len(classes)):
        sample_weights[train_labels == classes[i]] = weights[i]

    val_weights = np.ones(shape=val_labels.shape)
    for i in range(len(classes)):
        val_weights[val_labels == classes[i]] = weights[i]

    test_weights = np.ones(shape=test_labels.shape)
    for i in range(len(classes)):
        test_weights[test_labels == classes[i]] = weights[i]

    image_generator = image_datagen.flow(
        train_data,
        seed=seed)
    mask_generator = mask_datagen.flow(
        train_labels,
        sample_weight = sample_weights,
        seed=seed)

    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask_squeezed = np.squeeze(mask)
            mask_one_hot = tf.one_hot(mask_squeezed, depth=17, dtype=tf.int8)
            img = tf.convert_to_tensor(img)
            yield img, mask_one_hot

    generator = image_mask_generator(image_generator, mask_generator)

    # One-hot encoding the validation labels
    val_labels = tf.one_hot(val_labels, depth=17, dtype=tf.int8)

    ## -- Training the Unet Model --

    # Image shape
    img_rows = 32
    img_cols = 32
    img_channels = test_data.shape[3]

    #Output
    nb_classes = 17

    # Architecture Parameters
    nb_filters_0 = 32

    # Saving the model at each step
    checkpoint_filepath = "tmp/checkpoint_salinas_unet"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

    # Deep Learning Model
    model = Unet4(shape=(32, 32, test_data.shape[3]), nb_filters=32, kernel_size=3, initialization="glorot_uniform", output_channels=17, drop=0.2, regularization=l2(0.001205))
    
    
    print(model.summary())
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=0.001825), metrics=["accuracy"])

    history = model.fit(generator, batch_size=10, 
                steps_per_epoch=6, 
                epochs=200,
                validation_data = (val_data, val_labels, val_weights),
                callbacks=[model_checkpoint_callback, ES])
    
    np.save("logs/metrics/salinas/u_net/train_accuracy", history.history['accuracy'])
    np.save("logs/metrics/salinas/u_net/val_accuracy", history.history['val_accuracy'])

    pred = model.predict(test_data)
    print(pred.shape, test_labels.shape)
    print(np.argmax(pred[:, :, :, 1:], axis=3).shape)
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(test_labels, np.argmax(pred[:, :, :, 1:], axis=3)+1, sample_weight=test_weights)

    images = np.argmax(pred[:, :, :, 1:], axis=3)+1
    images[test_labels==0] = 0
    import matplotlib.pyplot as plt
    c = 0
    
    cmap = plt.get_cmap('viridis', 16)
    for image in images:
        plt.imshow(image, vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/u_net/pred{c}")
        plt.clf()

        plt.imshow(test_labels[c], vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/u_net/GT{c}")
        plt.clf()
        c+=1
 
    pred = np.argmax(pred[:, :, :, 1:], axis=3)+1
    pred = pred.flatten()
    test_labels = test_labels.flatten()
    print(pred.shape, test_labels.shape)
    conf = confusion_matrix(pred, test_labels)
    print(conf)
    np.save("logs/metrics/salinas/u_net/confusion_matrix", conf)
    print("Test accuracy = ", accuracy.result().numpy())

def u_net_sep_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights):
    # Creating an image generator
    image_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    mask_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    seed = 1
    l, x, y = train_labels.shape
    new_shape = (l, x, y, 1)
    train_labels = np.reshape(train_labels, new_shape)

    image_datagen.fit(train_data, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    sample_weights = np.ones(shape=train_labels.shape)
    for i in range(len(classes)):
        sample_weights[train_labels == classes[i]] = weights[i]

    val_weights = np.ones(shape=val_labels.shape)
    for i in range(len(classes)):
        val_weights[val_labels == classes[i]] = weights[i]

    test_weights = np.ones(shape=test_labels.shape)
    for i in range(len(classes)):
        test_weights[test_labels == classes[i]] = weights[i]

    image_generator = image_datagen.flow(
        train_data,
        seed=seed)
    mask_generator = mask_datagen.flow(
        train_labels,
        sample_weight = sample_weights,
        seed=seed)

    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            mask_squeezed = np.squeeze(mask)
            mask_one_hot = tf.one_hot(mask_squeezed, depth=17, dtype=tf.int8)
            img = tf.convert_to_tensor(img)
            yield img, mask_one_hot

    generator = image_mask_generator(image_generator, mask_generator)

    # One-hot encoding the validation labels
    val_labels = tf.one_hot(val_labels, depth=17, dtype=tf.int8)

    ## -- Training the Unet Model --

    # Image shape
    img_rows = 32
    img_cols = 32
    img_channels = test_data.shape[3]

    #Output
    nb_classes = 17

    # Architecture Parameters
    nb_filters_0 = 32

    # Saving the model at each step
    checkpoint_filepath = "tmp/checkpoint_salinas_unet"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)


    # Deep Learning Model
    model = UnetSeparable(shape=(32, 32, test_data.shape[3]), depth=test_data.shape[3], nb_filters=2, kernel_size=3, initialization="glorot_uniform", output_channels=17, drop=0.1, regularization=l2(0.0001))
    
    #model = Unet((img_rows, img_cols, img_channels), nb_filters=128, output_channels=nb_classes, initialization="he_normal", kernel_size=3, drop=0.50)
    print(model.summary())
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=0.0020259), metrics=["accuracy"])

    history = model.fit(generator, batch_size=5, 
                steps_per_epoch=50, 
                epochs=50,
                validation_data = (val_data, val_labels, val_weights),
                callbacks=[model_checkpoint_callback, ES])
    
    np.save("logs/metrics/salinas/u_net_sep/train_accuracy", history.history['accuracy'])
    np.save("logs/metrics/salinas/u_net_sep/val_accuracy", history.history['val_accuracy'])

    pred = model.predict(test_data)
    accuracy = tf.keras.metrics.Accuracy()

    accuracy.update_state(test_labels, np.argmax(pred[:, :, :, 1:], axis=3)+1, sample_weight=test_weights)

    images = np.argmax(pred[:, :, :, 1:], axis=3)+1
    images[test_labels==0] = 0
    import matplotlib.pyplot as plt
    c = 0
    cmap = plt.get_cmap('viridis', 16)
    for image in images:
        plt.imshow(image, vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/u_net_sep/pred{c}")
        plt.clf()

        plt.imshow(test_labels[c], vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/u_net_sep/GT{c}")
        plt.clf()
        c+=1
 
    pred = np.argmax(pred[:, :, :, 1:], axis=3)+1
    pred = pred.flatten()
    test_labels = test_labels.flatten()
    print(pred.shape, test_labels.shape)
    conf = confusion_matrix(pred, test_labels)
    print(conf)
    np.save("logs/metrics/salinas/u_net_sep/confusion_matrix", conf)
    print("Test accuracy = ", accuracy.result().numpy())

def cnn_1d_pipeline(train_data, train_labels, val_data, val_labels, test_data, test_labels, classes, weights): 

    
    
    train_data = train_data.reshape(train_data.shape[0]*train_data.shape[1]*train_data.shape[2], train_data.shape[3])
    train_labels = train_labels.reshape(train_labels.shape[0]*train_labels.shape[1]*train_labels.shape[2])

    val_data = val_data.reshape(val_data.shape[0]*val_data.shape[1]*val_data.shape[2], val_data.shape[3])
    val_labels = val_labels.reshape(val_labels.shape[0]*val_labels.shape[1]*val_labels.shape[2])

    test_shape = test_labels.shape
    test_data = test_data.reshape(test_data.shape[0]*test_data.shape[1]*test_data.shape[2], test_data.shape[3])
    test_labels = test_labels.reshape(test_labels.shape[0]*test_labels.shape[1]*test_labels.shape[2])    

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

    train_labels = tf.one_hot(train_labels, depth=17, dtype=tf.int8).numpy()
    val_labels = tf.one_hot(val_labels, depth=17, dtype=tf.int8).numpy()


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

    batch_size = 3000 
    steps_per_epoch = 20
    train_generator = DataGenerator(train_data, train_labels, train_weights, batch_size, shuffle=True)
    val_generator = DataGenerator(val_data, val_labels, val_weights, batch_size, shuffle=True)
    print(train_data.shape, train_labels.shape, train_weights.shape)

    checkpoint_filepath = "tmp/checkpoint_salinas_convnet"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model = cnn_1D(shape=(train_data.shape[1], 1), kernel_size=7, nb_filters_0 = 64, nb_dense_neurons=500, output_channels=17, kernel_reg=0.0009875, dense_reg=0.0012559, dropout=0)
    print(model.summary())

    print("model defined")
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=0.00074582), metrics=["accuracy"])
    print("model compiled")
    print(model.summary())
    print(train_data.shape)
    history = model.fit(train_generator, 
                steps_per_epoch=steps_per_epoch, 
                epochs=500,
                validation_data = val_generator,
                callbacks=[model_checkpoint_callback])
                
    np.save("logs/metrics/salinas/conv_net/train_accuracy", history.history['accuracy'])
    np.save("logs/metrics/salinas/conv_net/val_accuracy", history.history['val_accuracy'])


    y_pred = model.predict(test_data, batch_size=batch_size)

    y_pred = np.argmax(y_pred[:,1:], axis=1)+1
    
    accuracy = Accuracy()
    accuracy.update_state(test_labels, y_pred, sample_weight=test_weights)
    pred_accuracy = accuracy.result()
    print(pred_accuracy)

    print("Computing confusion matrix")
    conf = confusion_matrix(test_labels, y_pred)
    print(conf)
    np.save("logs/metrics/salinas/conv_net/confusion_matrix", conf)


    y_pred = y_pred.reshape(test_shape)
    print(y_pred.shape)
    y_pred[test_labels.reshape(test_shape)==0] = 0
    print("Saving raw predictions")
    import matplotlib.pyplot as plt
    c = 0
    cmap = plt.get_cmap('viridis', 16)
    for image in y_pred:
        plt.imshow(image, vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/cnn_1D/pred{c}_raw")
        plt.clf()

        plt.imshow(test_labels.reshape(test_shape)[c], vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/cnn_1D/GT{c}")
        plt.clf()

        c+=1

    for i in range(len(y_pred)):
        y_pred[i] = medfilt(y_pred[i], kernel_size = 3)

    print("Saving median filtered predictions")
    c = 0
    for image in y_pred:
        plt.imshow(image, vmin=0, vmax=16, cmap=cmap)
        plt.colorbar()
        plt.savefig(f"images/salinas/cnn_1D/pred{c}_filtered")
        plt.clf()

        c+=1
    
    y_pred = y_pred.reshape(test_shape[0]*test_shape[1]*test_shape[2])
    accuracy = Accuracy()
    accuracy.update_state(test_labels, y_pred, test_weights)
    pred_accuracy = accuracy.result()
    print(pred_accuracy)
