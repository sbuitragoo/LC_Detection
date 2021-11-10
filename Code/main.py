from data_aug import readImgs, img2tensor, GImg, randomBr, randomCont, randomSat, rgb2gray, crop
from get_bndbx import leer_annotations, get_points, img_conv, bndbox_img, aug_boxes
from network import VGG

import xml.etree.ElementTree as ET
from pathlib import Path
import os
import tensorflow as tf
import cv2
import numpy as np


xml_dir = '/Data/Annotations'
img_dir = '/Data/JPEGImages'
labels = ['Luisillo El Pillo']

train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)

boxes = get_points(train_imgs)

images = readImgs(img_dir)

images = GImg(images)

aug = randomBr(images)
images.append(aug)

aug = randomCont(images)
images.append(aug)

aug = randomSat(images)
images.append(aug)

aug = crop(images)
images.append(aug)

all_boxes = aug_boxes(boxes, 3)

model = VGG()