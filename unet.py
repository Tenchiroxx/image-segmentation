import tensorflow as tf

# -- Throtling GPU use -- 
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization, Activation, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def ConvBlock(tensor, nb_filters, kernel_size=3, padding='same', initializer='he_normal', activation="relu", regularization=None):
    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=regularization)(tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=regularization)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def DeconvBlock(tensor, residual, nb_filters, kernel_size=3, padding="same", strides=(2,2), regularization=None):
    y = Conv2DTranspose(nb_filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = ConvBlock(y, nb_filters, kernel_size, regularization=regularization)
    return y

def Unet(shape, nb_filters=32, exp=1, kernel_size=3, initialization="glorot_uniform", activation="relu", sigma_noise=0, output_channels=1, drop=0.0):
    
    input_layer = Input(shape=shape)

    conv1 = ConvBlock(input_layer, nb_filters=nb_filters, kernel_size=kernel_size, initializer=initialization, activation=activation )
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

    conv5 = ConvBlock(pool4, nb_filters=nb_filters * 2 **(4 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation )
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    if drop > 0.0: pool5 = Dropout(drop)(pool5)

    deconv6 = DeconvBlock(conv5, residual=conv4, nb_filters=nb_filters * 2 **(3 * exp), kernel_size=kernel_size)
    if drop > 0.0: deconv6 = Dropout(drop)(deconv6)

    deconv7 = DeconvBlock(deconv6, residual=conv3, nb_filters=nb_filters * 2 **(2 * exp), kernel_size=kernel_size)
    if drop > 0.0: deconv7 = Dropout(drop)(deconv7)

    deconv8 = DeconvBlock(deconv7, residual=conv2, nb_filters=nb_filters * 2 **(1 * exp), kernel_size=kernel_size)
    if drop > 0.0: deconv8 = Dropout(drop)(deconv8)

    deconv9 = DeconvBlock(deconv8, residual=conv1, nb_filters=nb_filters, kernel_size=kernel_size)
    if drop > 0.0: deconv9 = Dropout(drop)(deconv9)

    if sigma_noise > 0:
        deconv9 = GaussianNoise(sigma_noise)(deconv9)

    output_layer = Conv2D(filters=output_channels, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)


    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model


def Unet4(shape, nb_filters=32, exp=1, kernel_size=3, initialization="glorot_uniform", activation="relu", sigma_noise=0, output_channels=1, drop=0.0, regularization=None):
    input_layer = Input(shape=shape)

    conv1 = ConvBlock(input_layer, nb_filters=nb_filters, kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if drop > 0.0: pool1 = Dropout(drop)(pool1)

    conv2 = ConvBlock(pool1, nb_filters=nb_filters * 2 **(1 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if drop > 0.0: pool2 = Dropout(drop)(pool2)

    conv3 = ConvBlock(pool2, nb_filters=nb_filters * 2 **(2 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if drop > 0.0: pool3 = Dropout(drop)(pool3)

    conv4 = ConvBlock(pool3, nb_filters=nb_filters * 2 **(3 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if drop > 0.0: pool4 = Dropout(drop)(pool4)

    

    deconv5 = DeconvBlock(conv4, residual=conv3, nb_filters=nb_filters * 2 **(2 * exp), kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv5 = Dropout(drop)(deconv5)

    deconv6 = DeconvBlock(deconv5, residual=conv2, nb_filters=nb_filters * 2 **(1 * exp), kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv6 = Dropout(drop)(deconv6)

    deconv7 = DeconvBlock(deconv6, residual=conv1, nb_filters=nb_filters, kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv7 = Dropout(drop)(deconv7)



    if sigma_noise > 0:
        deconv7 = GaussianNoise(sigma_noise)(deconv7)

    output_layer = Conv2D(filters=output_channels, kernel_size=(1, 1))(deconv7)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)


    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model


def ConvBlockSeparable(tensor, nb_filters, depth, kernel_size=3, padding='same', initializer='he_normal', activation="relu", regularization=None):
    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, groups=depth, kernel_regularizer=regularization)(tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, groups=depth, kernel_regularizer=regularization)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def UnetSeparable(shape, depth, nb_filters=4, exp=1, kernel_size=5, initialization="glorot_uniform", activation="relu", sigma_noise=0, output_channels=1, drop=0.3, regularization=0.00013443):
    
    input_layer = Input(shape=shape)

    conv1 = ConvBlockSeparable(input_layer, depth=depth, nb_filters=depth*nb_filters, kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if drop > 0.0: pool1 = Dropout(drop)(pool1)

    conv2 = ConvBlockSeparable(pool1, depth=depth, nb_filters=depth*nb_filters * 2 **(1 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if drop > 0.0: pool2 = Dropout(drop)(pool2)

    conv3 = ConvBlockSeparable(pool2, depth=depth, nb_filters=depth*nb_filters * 2 **(2 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if drop > 0.0: pool3 = Dropout(drop)(pool3)

    conv4 = ConvBlockSeparable(pool3, depth=depth, nb_filters=depth*nb_filters * 2 **(3 * exp), kernel_size=kernel_size, initializer=initialization, activation=activation, regularization=regularization)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if drop > 0.0: pool4 = Dropout(drop)(pool4)


    deconv6 = DeconvBlock(pool4, residual=pool3, nb_filters=depth*nb_filters * 2 **(3 * exp), kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv6 = Dropout(drop)(deconv6)

    deconv7 = DeconvBlock(deconv6, residual=conv3, nb_filters=depth*nb_filters * 2 **(2 * exp), kernel_size=kernel_size,regularization=regularization)
    if drop > 0.0: deconv7 = Dropout(drop)(deconv7)

    deconv8 = DeconvBlock(deconv7, residual=conv2, nb_filters=depth*nb_filters * 2 **(1 * exp), kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv8 = Dropout(drop)(deconv8)

    deconv9 = DeconvBlock(deconv8, residual=conv1, nb_filters=depth*nb_filters, kernel_size=kernel_size, regularization=regularization)
    if drop > 0.0: deconv9 = Dropout(drop)(deconv9)

    if sigma_noise > 0:
        deconv9 = GaussianNoise(sigma_noise)(deconv9)

    output_layer = Conv2D(filters=output_channels, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)


    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

def cnn_1D(shape, kernel_size, nb_filters_0, nb_dense_neurons, kernel_reg, dense_reg, output_channels=1, dropout=0.0, learning_rate=1e-3):
    
    kernel_initializer = "glorot_uniform"
    input_layer = Input(shape=shape)

    x = Conv1D(filters=nb_filters_0, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=None, padding='valid')(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=nb_filters_0*2, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)

    # x = Conv1D(filters=nb_filters_0*4, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_reg), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # # x = MaxPooling1D(pool_size=2)(x)
    # x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(nb_dense_neurons, kernel_regularizer=None)(x)

    x = Dropout(dropout)(x)
    

    output_layer = Dense(output_channels, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

    return model
    


    
