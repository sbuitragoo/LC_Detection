import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Concatenate, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

def resize_img(img, size):
    img = tf.image.resize(img, (size, size))
    return img


def Conv(x, filt, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1,0),(1,0)))(x)
        padding = 'valid'
    
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding, use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return x


def Residual(x, filters):
    prev = x
    x = Conv(x, filters // 2, 1)
    x = Conv(x, filters, 3)
    x = Add()([prev, x])
    return x


def Block(x, filters, blocks):
    x = Conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = Residual(x, filters)
    return x

def VGG():
    m_input = Input(shape=(224,224,3))
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(m_input)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2,strides=2, padding='same')(x)

    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    x = Flatten()(x) 
    x = Dense(units = 4096, activation ='relu')(x) 
    x = Dense(units = 4096, activation ='relu')(x) 
    output = Dense(units = 1000, activation ='softmax')(x)


    model = Model(inputs=m_input, outputs= output)

    return model
