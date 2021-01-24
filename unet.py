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
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization, Activation, Conv2DTranspose


def u_net(shape, nb_filters_0=32, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", sigma_noise=0, output_channels=1, drop=0.0):
    """U-Net model.

    Standard U-Net model, plus optional gaussian noise.
    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.
    drop: dropout rate

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this function comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = Input(shape)
    print(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_1")(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if drop > 0.0: pool1 = Dropout(drop)(pool1)

    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if drop > 0.0: pool2 = Dropout(drop)(pool2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if drop > 0.0: pool3 = Dropout(drop)(pool3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if drop > 0.0: pool4 = Dropout(drop)(pool4)

    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(pool4)
    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)
    if drop > 0.0: conv5 = Dropout(drop)(conv5)

    up6 = concatenate(
        [UpSampling2D(size=(2, 2))(conv5), conv4], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)
    if drop > 0.0: conv6 = Dropout(drop)(conv6)

    up7 = concatenate(
        [UpSampling2D(size=(2, 2))(conv6), conv3], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)
    if drop > 0.0: conv7 = Dropout(drop)(conv7)

    up8 = concatenate(
        [UpSampling2D(size=(2, 2))(conv7), conv2], axis=channel_axis)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_1")(up8)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_2")(conv8)
    if drop > 0.0: conv8 = Dropout(drop)(conv8)

    up9 = concatenate(
        [UpSampling2D(size=(2, 2))(conv8), conv1], axis=channel_axis)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_1")(up9)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_2")(conv9)
    if drop > 0.0: conv9 = Dropout(drop)(conv9)

    if sigma_noise > 0:
        conv9 = GaussianNoise(sigma_noise)(conv9)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv9)

    return Model(inputs, conv10)


def u_net3(shape, nb_filters_0=32, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", sigma_noise=0, output_channels=1):
    """U-Net model, with three layers.

    U-Net model using 3 maxpoolings/upsamplings, plus optional gaussian noise.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this function comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = Input(shape)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_1")(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)

    up5 = concatenate(
        [UpSampling2D(size=(2, 2))(conv4), conv3], axis=channel_axis)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(up5)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)

    up6 = concatenate(
        [UpSampling2D(size=(2, 2))(conv5), conv2], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)

    up7 = concatenate(
        [UpSampling2D(size=(2, 2))(conv6), conv1], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)

    if sigma_noise > 0:
        conv7 = GaussianNoise(sigma_noise)(conv7)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv7)

    return Model(inputs, conv10)




# -- Another Unet implementation

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def unet4(img_height, img_width, img_depth, nclasses=3, filters=64):
# down
    input_layer = Input(shape=(img_height, img_width, img_depth), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model