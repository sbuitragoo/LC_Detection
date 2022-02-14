import xml.etree.ElementTree as ET
from pathlib import Path
import os
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#-------------------------------Etiquetas-------------------------------#

def leer_annotations(ann_dir, img_dir, labels=[]):
    """Gets bounding box location for each image

    Args:
        ann_dir ([string]): [Annotations path]
        img_dir ([string]): [Images path]
        labels (list, optional): [description]. Defaults to [].

    Returns:
        [Dict]: [Dictionary with label's information]
    """

    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
 
        tree = ET.parse(os.path.join(ann_dir, ann))
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
              
                        img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
 
        if len(img['object']) > 0:
          all_imgs+=[img]
                        
    return all_imgs, seen_labels


def get_points(train_imgs):
    """Saves bndbox corners inside an array

    Args:
        train_imgs ([dict]): [Dictionary with image bndbox info]
    """
    cordinates = np.zeros((len(train_imgs),4))
    for i in range(len(train_imgs)):
        j = 0
        cordinates[i,j] = train_imgs[i]['object'][0]['xmax']
        cordinates[i,j+1] = train_imgs[i]['object'][0]['xmin']
        cordinates[i,j+2] = train_imgs[i]['object'][0]['ymax']
        cordinates[i,j+3] = train_imgs[i]['object'][0]['ymin']
    return cordinates

def flip_boxes(boxes):
    orig = boxes
    for i in range(len(boxes)):
        j = 0
        boxes[i,j] = orig[i,j+3] #Xmax se cambia por Ymin
        boxes[i,j+1] = orig[i,j+2] #Xmin se cambia por Ymax
        boxes[i,j+2] = orig[i,j] #Ymax se cambia por Xmax
        boxes[i,j+3] = orig[i,j+1] #Ymin se cambia por Xmin
    return boxes

def reScale_bndboxes(shapes, bndbox, x):
    for i in range(len(bndbox)):
        j=0
        xscale = x/shapes[i,j]
        yscale = x/shapes[i,j+1]
        bndbox[i,j] =  bndbox[i,j]*xscale #xmax
        bndbox[i,j+1] =  bndbox[i,j+1]*xscale #xmin
        bndbox[i,j+2] =  bndbox[i,j+2]*yscale #ymax
        bndbox[i,j+3] =  bndbox[i,j+3]*yscale #ymin
    return bndbox

def AugmentBoxes(images, boxes):
    new_boxes = np.zeros((images.shape[0], 4))
    fil = 0
    for i in np.arange(0,len(new_boxes),4):
        for j in range(4):
            new_boxes[i,j] = boxes[fil,j]
            new_boxes[i+1,j] = boxes[fil,j]
            new_boxes[i+2,j] = boxes[fil,j]
            new_boxes[i+3,j] = boxes[fil,j]
        fil+=1
    return new_boxes

#-----------------------------Datos-----------------------------#


def PreProcessImages(img_path):
  """Read input images

  Args:
      img_path (str): image path

  Returns:
      list: List with cv2 images
  """
  path = sorted(os.listdir(img_path))
  imgs = []
  shape = np.zeros((len(path), 2))
  for i in range(len(path)):
    j = 0
    img = cv2.imread(os.path.join(img_path, path[i]))
    shape[i,j] = img.shape[0]
    shape[i,j+1] = img.shape[1]
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img/255.0
    imgs.append(img)
    j+=1
  return imgs, shape

def DataAugmentation(images):
    augmented_data = []
    for i in range(len(images)):
        seed = (i,0)
        m1 = tf.image.stateless_random_brightness(
                    images[i] ,max_delta=0.4, seed=seed)
        m2 = tf.image.stateless_random_saturation(images[i], 0.5, 1.0, seed)
        m3 = tf.image.stateless_random_contrast(images[i],0.2,0.5,seed=seed)
        augmented_data.append(images[i])
        augmented_data.append(m1)
        augmented_data.append(m2)
        augmented_data.append(m3)
    return augmented_data


#-----------------------------------Process---------------------------------#


def data(xml_dir, img_dir, labels):

    train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)

    images, shapes = PreProcessImages(img_dir)

    boxes = get_points(train_imgs)
    boxes = reScale_bndboxes(shapes, boxes, 224)

    full_data = DataAugmentation(images)
    full_data = np.array(full_data)
    full_labels = AugmentBoxes(full_data, boxes)

    train_images, test_images, train_targets, test_targets = train_test_split(full_data, full_labels, test_size=0.2)

    return train_images, test_images, train_targets, test_targets
