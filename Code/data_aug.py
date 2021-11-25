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
  for i in range(len(path)):
    img = cv2.imread(img_path + path[i])
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    imgs.append(img)
  return imgs


def img2tensor(arg):
  """Convert Cv2 images to tensors

  Args:
      arg ([cv2 image]): []

  Returns:
      [tf.tensor]: [image tensor]
  """
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  arg = arg/255.0
  return arg


def net_img(imgs):
  """Transform a list of images into a list of tensors

  Args:
      imgs ([list]): [list with images]

  Returns:
      [list]: [list with image tensors]
  """
  Tens = []
  for i in range(len(imgs)):
    im = imgs[i]
    tensor = img2tensor(im)
    Tens.append(tensor)
  return Tens


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
        flip_left_right = t.image.flip_left_right(imgs[i])
        flipped.append(flip_left_right)
    return flipped

def transpose(imgs):
    flipped = []
    for i in range(len(imgs)):
        transpose_image = tf.image.transpose_image(imgs[i])
        flipped.append(transpose_image)
    return flipped


def randomSat(imgs):
    """Change Saturation of a set of images

    Args:
        imgs ([list]): [list of images to modify]

    Returns:
        [list]: [images with new saturation]
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

def data(xml_dir, img_dir, labels):

    #xml_dir = './Data/Annotations/'
    #img_dir = './Data/JPEGImages/'
    #labels = ['Luisillo El Pillo']

    train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)

    boxes = get_points(train_imgs)

    images = readImgs(img_dir)

    images = net_img(images)

    train_images, test_images ,train_targets, test_targets = train_test_split(images,boxes, test_size=0.2)

    #---------------------Train_Data Augmentation---------------------#

    ImagesToAugmentTrain = train_images
    
    aug = randomBr(ImagesToAugmentTrain)
    train_images.append(aug)

    aug = randomCont(ImagesToAugmentTrain)
    train_images.append(aug)

    aug = randomSat(ImagesToAugmentTrain)
    train_images.append(aug)

    aug = rgb2gray(ImagesToAugmentTrain)
    train_images.append(aug)

    aug = transpose(ImagesToAugmentTrain)
    train_images.append(aug)

    #---------------------Train Boxes---------------------#

    train_transposed_targets = flip_boxes(train_targets)
    train_targets = aug_boxes(train_targets, 3)
    train_targets = np.concatenate((train_targets, train_transposed_targets))


    #---------------------Test_Data Augmentation---------------------#

    ImagesToAugmentTest = test_images

    aug = randomBr(ImagesToAugmentTest)
    test_images.append(aug)

    aug = randomCont(ImagesToAugmentTest)
    test_images.append(aug)

    aug = randomSat(ImagesToAugmentTest)
    test_images.append(aug)

    aug = rgb2gray(ImagesToAugmentTest)
    test_images.append(aug)

    aug = transpose(ImagesToAugmentTest)
    test_images.append(aug)

    #---------------------Test Boxes---------------------#

    test_transposed_targets = flip_boxes(test_targets)
    test_targets = aug_boxes(test_targets, 3)
    test_targets = np.concatenate((test_targets, test_transposed_targets))

    return train_images, test_images, train_targets, test_targets
