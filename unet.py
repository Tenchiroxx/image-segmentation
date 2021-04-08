"""
Utility that contains all the neural networks used for the project:
- U-Net
- U-Net Separable
- 1D CNN
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization, Activation, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

##################################
############# U-NET ##############
##################################

def ConvBlock(tensor, nb_filters, kernel_size=3, padding='same', initializer='he_normal', activation="relu", regularization=None):

    """
    Base convolution block used in the encoding part of the U-Net

    Parameters :
    ------------

    tensor : tf.Tensor
    Output of the previous layer

    nb_filters : int
    Number of filters in the first layer of the U-Net

    kernel_size : int
    Size of the convolution kernel

    padding : str
    Convolution padding used on the edges 

    initializer : str
    Initializer used for the weights of the layers

    activation : str
    Activation function used on the output of the convolution layer

    Returns :
    -----------

    x : tf.Tensor
    Output layer 
    """


    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=regularization)(tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=nb_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer, kernel_regularizer=regularization)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def DeconvBlock(tensor, residual, nb_filters, kernel_size=3, padding="same", strides=(2,2), regularization=None):

    """
    Base convolution block used in the decoding part of the U-Net

    Parameters :
    ------------

    tensor : tf.Tensor
    Output of the previous layer

    residual : tf.Tensor
    Output of the equivalent layer of the encoding part of the U-Net

    nb_filters : int
    Number of filters in the first layer of the U-Net

    kernel_size : int
    Size of the convolution kernel

    padding : str
    Convolution padding used on the edges 

    strides : tuple
    Strides made by the convolution kernel

    Returns :
    -----------

    x : tf.Tensor
    Output layer 
    
    """
    y = Conv2DTranspose(nb_filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = ConvBlock(y, nb_filters, kernel_size, regularization=regularization)
    return y

def Unet4(shape, nb_filters=32, exp=1, kernel_size=3, initialization="glorot_uniform", activation="relu", sigma_noise=0, output_channels=1, drop=0.0, regularization=None):
    
    """
    U-Net architecture with 4 convolution blocks in the encoding part. 

    Parameters :
    ------------

    shape : tuple
    (x, y, z), shape of the input image.

    nb_filters : int
    Number of filters in the first layer of the U-Net

    kernel_size : int
    Size of the convolution kernel

    exp : int
    Exponential multiplication factor for the number of layers in the successive layers of the U-Net. 

    initialization : str
    Initializer used for the weights of the layers

    activation : str
    Activation function used on the output of the convolution layer

    sigma_noise : float
    Noise used in the network (helps for regularization)

    output_channels : int
    Number of channels in the output layer

    Returns :
    -----------

    x : tf.Tensor
    Output layer 
    """
    
    
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


##################################
######## SEPARABLE U-NET #########
##################################

def ConvBlockSeparable(tensor, nb_filters, depth, kernel_size=3, padding='same', initializer='he_normal', activation="relu", regularization=None):

    """
    Base convolution block used in the encoding part of the separable U-Net

    Parameters :
    ------------

    tensor : tf.Tensor
    Output of the previous layer

    nb_filters : int
    Number of filters in the first layer of the U-Net

    depth : int
    Number of channels in the input image

    kernel_size : int
    Size of the convolution kernel

    padding : str
    Convolution padding used on the edges 

    initializer : str
    Initializer used for the weights of the layers

    activation : str
    Activation function used on the output of the convolution layer

    Returns :
    -----------

    x : tf.Tensor
    Output layer 
    """

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



##################################
############# 1D CNN #############
##################################

def cnn_1D(shape, kernel_size, nb_filters_0, nb_dense_neurons, kernel_reg, dense_reg, output_channels=1, dropout=0.0, learning_rate=1e-3):
    
    """
    1D CNN architecture

    Parameters :
    ------------

    shape : tuple
    Shape of the input pixel

    kernel_size : int
    Size of the convolution kernel

    nb_filters_0 : int
    Number of filters in the first layer of the 1D CNN.

    kernel_reg : function
    Regularization function used for the convolution layers

    dense_reg : function
    Regularization function used for the dense layers

    output_channels : int
    Number of channels in the output layer

    learning_rate : float
    Learning rate of the optimizer.

    Returns :
    -----------

    x : tf.Tensor
    Output layer 
    """


    kernel_initializer = "glorot_uniform"
    input_layer = Input(shape=shape)

    x = Conv1D(filters=nb_filters_0, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=None, padding='valid')(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=nb_filters_0*2, kernel_size=kernel_size, kernel_initializer=kernel_initializer, kernel_regularizer=None, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(nb_dense_neurons, kernel_regularizer=None)(x)
    x = Dropout(dropout)(x)
    
    output_layer = Dense(output_channels, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss = "categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

    return model
    


    
