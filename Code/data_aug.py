from sklearn.model_selection import train_test_split
from bndbx import leer_annotations, get_points, img_conv, bndbox_img, aug_boxes, reScale_bndboxes, flip_boxes
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import tensorflow as tf
import cv2
import numpy as np

def readImgs(img_path):
  """Read input images

  Args:
      img_path ([str]): [image path]

  Returns:
      [list]: [List with cv2 images]
  """
  path = sorted(os.listdir(img_path))
  imgs = []
  shape = np.zeros((len(path), 2))
  for i in range(len(path)):
    j = 0
    img = cv2.imread(img_path + path[i])
    shape[i,j] = img.shape[0]
    shape[i,j+1] = img.shape[1]
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    imgs.append(img)
    j+=1
  return imgs, shape


def randomBr(imgs):
    """Change de Brightness of a set of images

    Args:
        imgs ([list]): [list of images to modify]

    Returns:
        [list]: [list of images with brightness changed]
    """
    rb = []
    for i in range(len(imgs)):
        seed = (i, 0)
        srb = tf.image.stateless_random_brightness(
                imgs[i] ,max_delta=0.4, seed=seed)
        rb.append(srb)
    return rb

def flip_mirror(imgs):
    flipped = []
    for i in range(len(imgs)):
        flip_left_right = tf.image.flip_left_right(imgs[i])
        flipped.append(flip_left_right)
    return flipped

def transpose(imgs):
    """Transposes a set of images

    Args:
        imgs (list): Set of images to transpose

    Returns:
        list: Transposed images
    """
    transposed = []
    for i in range(len(imgs)):
        transpose_image = tf.image.transpose(imgs[i])
        transposed.append(transpose_image)
    return transposed


def randomSat(imgs):
    """Change Saturation of a set of images

    Args:
        imgs (list): list of images to modify

    Returns:
        list: images with new saturation
    """
    sat = []
    for i in range(len(imgs)):
        seed = (i,0)
        satu = tf.image.stateless_random_saturation(imgs[i], 0.5, 1.0, seed)
        sat.append(satu)
    return sat


def rgb2gray(imgs):
    """Change images to grayscale

    Args:
        imgs ([list]): [images to change their channels]

    Returns:
        [list]: [grayscale images]
    """
    gscale = []
    for i in range(len(imgs)):
        gray = tf.image.rgb_to_grayscale(imgs[i])
        gscale.append(gray)
    return gscale


def crop(imgs):
    """Crop a set of images

    Args:
        imgs ([list]): [images to crop]

    Returns:
        [list]: [list with cropped images]
    """
    crp = []
    for i in range(len(imgs)):
        cropped = tf.image.central_crop(imgs[i], central_fraction=0.5)
        crp.append(cropped)
    return crp


def randomCont(imgs):
    """Change contrast to a set of images

    Args:
        imgs ([list]): [images to modify]

    Returns:
        [list]: [images with new random contrast]
    """
    con = []
    for i in range(len(imgs)):
        seed = (i,0)
        cont = tf.image.stateless_random_contrast(imgs[i],0.2,0.5,seed=seed)
        con.append(cont)
    return con

