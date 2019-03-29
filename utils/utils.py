import os
from copy import deepcopy
import time
from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet
from keras.preprocessing import image
import cv2
from config import TRAIN_DIR, TEST_DIR, GREY_SCALE, USE_CROP, DATA_DIR


bboxes = pd.read_csv(os.path.join(DATA_DIR, 'bounding_boxes.csv'))


def preprocess_input(x, model_name):
    """
    Preprocess input in accordance to Keras preprocessing standards
    :param x: Numpy array with values between 0 and 255
    :param model_name: Name of NN model
    :return: Numpy array with preprocessed values
    """
    if model_name[:8] == 'resnet50':
        return preprocess_input_resnet50(x.copy())
    elif model_name[:5] == 'vgg19':
        return preprocess_input_vgg19(x.copy())
    elif model_name[:8] == 'xception':
        return preprocess_input_xception(x.copy())
    elif model_name[:6] == 'nasnet':
        return preprocess_input_nasnet(x.copy())
    elif model_name[:16] == 'inception_resnet':
        return preprocess_input_inception_resnet(x.copy())
    else:
        raise Exception('No such model preprocessing defined!')


def load_image(image_name, input_shape, mode, grayscale=GREY_SCALE, crop=USE_CROP):
    """
    Load image and convert to Numpy array with values between 0 and 255
    :param image_name: String with image name for TRAIN_DIR or TEST_DIR directory
    :param input_shape: 3-element tuple
    :return: Numpy array with values between 0 and 255 with shape self.train_conf.input_shape
    """
    if mode == "train":
        DIR = TRAIN_DIR
    elif mode == "test":
        DIR = TEST_DIR
    else:
        raise Exception("{} is not a valid mode!".format(mode))

    img = image.load_img(os.path.join(DIR, image_name), grayscale=grayscale)      
    img = image.img_to_array(img).astype(np.uint8)
    
    if grayscale:
        img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=-1)
               
    if crop:       
        coords = bboxes[bboxes['Image'] == image_name]
        
        x0 = coords.x0.values[0] 
        x1 = coords.x1.values[0] 
        y0 = coords.y0.values[0] 
        y1 = coords.y1.values[0] 
        x_diff = x1 - x0 
        y_diff = y1 - y0
        
        target_padding = 50
        pad_x = int(target_padding*(x_diff/input_shape[1]))
        pad_y = int(target_padding*(y_diff/input_shape[0]))
        
        img = img[max(y0-pad_y,0):y1+pad_y,max(x0-pad_x,0):x1+pad_x]     
        
    img = cv2.resize(img, (input_shape[0], input_shape[0])) 

    return img


def prepare_tensor_from_image(img, model_name):
    """
    Expand first dimension of input Numpy array:
    self.train_conf.input_shape -> (1, *self.train_conf.input_shape)
    And preprocces its values using Keras preprocessing
    :param img: Numpy array with values between 0 and 255 with shape self.train_conf.input_shape
    :param model_name: Name of NN model
    :return: Numpy array with shape (1, *self.train_conf.input_shape)
    """
    tensor = np.expand_dims(img, axis=0)
    preprocessed_tensor = preprocess_input(tensor, model_name)
    return preprocessed_tensor


def plot_side_by_side(imgs, titles=None):
    fig = plt.figure(figsize=(20, 20))
    columns = len(imgs)
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i, title=titles[i - 1] if titles is not None else None)
        try:
            img = imgs[i - 1]
        except IndexError:
            img = imgs[i - 1].squeeze(axis=2)
        plt.imshow(img)
    plt.show()


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        with Timer() as t:
            ret = fn(*args, **kwargs)

        return ret, t.secs

    return with_profiling
