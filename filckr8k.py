import tensorflow as tf
tf.enable_eager_execution()
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
import tensorflow as tf
tf.enable_eager_execution()


# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# extract descriptions for images
def load_descriptions(doc):
  mapping = dict()
  all_img_name_vector1 = []
  all_captions1 = []


  # process lines
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
      continue
    # take the first token as the image id, the rest as the description
    image_id, image_desc = tokens[0], tokens[1:]
    # remove filename from image id
    image_id = image_id.split('.')[0]
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    # create the list if needed
    if image_id not in mapping:
      mapping[image_id] = list()
    # store description
    mapping[image_id].append(image_desc)

    caption = '<start> ' + image_desc + ' <end>'
    if image_id ==  '2258277193_586949ec62':    
      pass
    else:
      full_coco_image_path = 'data/flickr8k/Flicker8k_Dataset/' + '%s.jpg' % (image_id)
      all_img_name_vector1.append(full_coco_image_path)
      all_captions1.append(caption)


  return all_captions1,all_img_name_vector1

def get_flickr8k_data():

    name_of_zip = 'Flickr8k_Dataset.zip'

    data_dir = os.path.join(os.path.abspath('.'), 'data/flickr8k')

    if not os.path.exists(data_dir + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip, 
                                          cache_subdir=data_dir,
                                          origin = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip',
                                          extract = True)
      PATH = os.path.dirname(image_zip)+'/Flickr8k_Dataset/'
    else:
      PATH = data_dir+'/Flickr8k_Dataset/'



    name_of_zip = 'Flickr8k_text.zip'

    data_dir = os.path.join(os.path.abspath('.'), 'data/flickr8k')

    if not os.path.exists(data_dir + '/' + name_of_zip):
      image_zip = tf.keras.utils.get_file(name_of_zip, 
                                          cache_subdir=data_dir,
                                          origin = 'http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip',
                                          extract = True)
      PATH = os.path.dirname(image_zip)+'/Flickr8k_text/'
    else:
      PATH = data_dir+'/Flickr8k_text/'

      import string


    all_captions1 = []
    all_img_name_vector1 = []
    filename = 'data/flickr8k/Flickr8k.token.txt'
    # load descriptions
    doc = load_doc(filename)
    # parse descriptions
    all_captions1,all_img_name_vector1 = load_descriptions(doc)

    train_caption=[]
    img_name_vector=[]
    train_captions = all_captions1
    img_name_vector = all_img_name_vector1
    print('Captions loaded:', len(train_captions) )
    print('Images:', len(img_name_vector) )
    return train_captions,img_name_vector
