import xml.etree.ElementTree as ET
from pathlib import Path
import os
import tensorflow as tf
import cv2
import numpy as np


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
 
        tree = ET.parse(ann_dir + ann)
        
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

def img_conv(imgr,cord,fila):
  path = sorted(os.listdir(imgr))
  img = cv2.imread('Detección/JPEGImages/' + path[fila])
  sp, ep = (int(cord[fila,1]), int(cord[fila,3])), (int(cord[fila,0]), int(cord[fila,2]))
  colors = (255,255,0)
  nimg = cv2.rectangle(img, sp, ep, colors, thickness = 2)
  return nimg