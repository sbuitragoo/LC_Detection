import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Concatenate, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from bndbx import get_points, leer_annotations, reScale_bndboxes, rezise_images
from data_aug import readImgs, net_img, train_images, test_images, train_targets, test_targets
from network import VGG

"""
xml_dir = './Data/Annotations/'
img_dir = './Data/JPEGImages/'
labels = ['Luisillo El Pillo']

train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)

boxes = get_points(train_imgs)

images = readImgs(img_dir)

images, boxes = rezise_data(images, boxes, 224)

images = np.asarray(net_img(images))

train_images, test_images, train_targets, test_targets = train_test_split(images,boxes, test_size=0.2)
"""

Xtrain = train_images
Xtest = test_images
ytrain = train_targets
ytest = test_targets

Xtrain = rezise_images(Xtrain, 224)
Xtrain = np.asarray(Xtrain)
ytrain = reScale_bndboxes(Xtrain, ytrain, 224)
Xtest = rezise_images(Xtest, 224)
Xtest = np.asarray(Xtest)
ytest = reScale_bndboxes(Xtest, ytest, 224)

model = VGG()

optimizer = Adam(1e-4)

model.compile(loss = 'mse', optimizer = optimizer, metrics=['accuracy'])

hitory = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), batch_size=16, epochs = 20, verbose=1)

#model.save('LC_detector.h5')