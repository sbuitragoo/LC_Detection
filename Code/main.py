#from data_aug import *
from network import VGG
from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from functions import data

xml_dir = '.\Data\Annotations'
img_dir = '.\Data\JPEGImages'
labels = ['Luisillo El Pillo']

Xtrain, Xtest, ytrain, ytest = data(xml_dir, img_dir, labels)

#print(Xtest[0].shape)
mod = tf.keras.models.load_model(".\LC_detector.h5")

y = mod.predict(np.expand_dims(Xtest[0], 0))
print(ytest[0], y)

img = Xtest[0]
sp, ep = (int(y[0,0]), int(y[0,2])), (int(y[0,1]), int(y[0,3]))
colors = (255,255,0)
nimg = cv2.rectangle(img, sp, ep, colors, thickness = 2)

plt.imshow(nimg)
plt.show()

