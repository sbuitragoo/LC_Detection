import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Concatenate, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from network import VGG
import cv2
from functions import data

xml_dir = '.\Data\Annotations'
img_dir = '.\Data\JPEGImages'
labels = ['Luisillo El Pillo']

Xtrain, Xtest, ytrain, ytest = data(xml_dir, img_dir, labels)


print("Entrenamiento: ", Xtrain.shape, ytrain.shape)
print("Test: ", Xtest.shape, ytest.shape)



# colors = (255,0,0)


# while True:
#     cv2.imshow('image',cv2.rectangle(img, sp, ep, colors, thickness = 2))
#     #cv2.keywait(500)

model = VGG()

optimizer = Adam(1e-4)

model.compile(loss = 'mse', optimizer = optimizer)

hitory = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), batch_size=4, epochs = 20, verbose=1)

#model.save('LC_detector.h5')