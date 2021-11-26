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
      imgs (list): list with images

  Returns:
      list: list with image tensors
  """
  tensores = []
  for i in range(len(imgs)):
    im = imgs[i]
    tensor = img2tensor(im)
    tensores.append(tensor)
  return tensores


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

def data(xml_dir, img_dir, labels):

    train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)

    images, shapes = readImgs(img_dir)

    print(shapes.shape)

    boxes = get_points(train_imgs)
    boxes = reScale_bndboxes(shapes, boxes, 224)

    images = net_img(images)

    train_images, test_images, train_targets, test_targets = train_test_split(images,boxes, test_size=0.2)

    #---------------------Train_Data Augmentation---------------------#

    ImagesToAugmentTrain = train_images.copy()
    
    aug1 = randomBr(ImagesToAugmentTrain)
    train_images.append(aug1)

    aug2 = randomCont(ImagesToAugmentTrain)
    train_images.append(aug2)

    aug3 = randomSat(ImagesToAugmentTrain)
    train_images.append(aug3)

    aug4 = transpose(train_images)
    train_images.append(aug4)

    #---------------------Train Boxes---------------------#

    train_transposed_targets = flip_boxes(train_targets)
    train_transposed_targets = aug_boxes(train_transposed_targets, 3)
    train_targets = aug_boxes(train_targets, 3)
    train_targets = np.concatenate((train_targets, train_transposed_targets))


    #---------------------Test_Data Augmentation---------------------#

    ImagesToAugmentTest = test_images.copy()

    aug5 = randomBr(ImagesToAugmentTest)
    test_images.append(aug5)

    aug6 = randomCont(ImagesToAugmentTest)
    test_images.append(aug6)

    aug7 = randomSat(ImagesToAugmentTest)
    test_images.append(aug7)

    aug8 = transpose(test_images)
    test_images.append(aug8)

    #---------------------Test Boxes---------------------#

    test_transposed_targets = flip_boxes(test_targets)
    test_transposed_targets = aug_boxes(test_transposed_targets, 3)
    test_targets = aug_boxes(test_targets, 3)
    test_targets = np.concatenate((test_targets, test_transposed_targets))

    return train_images, test_images, train_targets, test_targets
