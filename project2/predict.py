import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse
import datetime
import os
import random
import re
import sys

from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image
from skimage.io import imread, imshow
from skimage.transform import resize
import sklearn.metrics
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from time import strftime
from tqdm import tqdm

# Image definition
IMG_WIDTH    = 400
IMG_HEIGHT   = 400
IMG_CHANNELS = 3
PIXEL_DEPTH  = 255

# Folder definitions
IMAGE_DATA_PATH           = 'training/images/'
MASK_DATA_PATH            = 'training/groundtruth/'
MODEL_SAVE_LOCATION       = 'road_segmentation_model.h5'
…os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed = SEED
tf.random.set_seed(SEED)

# Checkpoints
checkpoint_path = 'checkpoints/cp.ckpt'
checkpoint_dir  = os.path.dirname(checkpoint_path)

# Image generation
GENERATE_NEW_IMG   = False
USE_GENERATED_IMG  = False
IMG_TO_GEN_PER_IMG = 100

# Load existing model
USE_SAVED_MODEL = True

# F1-score estimation
NUMBER_OF_IMG_TO_TEST = 10

# Predictions
RUN_PREDICTIONS_ON_TEST_IMAGES = True

if(USE_GENERATED_IMG):
    print("[INFO]: Updating images_filename")
    IMAGE_DATA_PATH = OUTPUT_DATA_IMAGE_PATH+'images/'
    MASK_DATA_PATH = OUTPUT_DATA_IMAGE_PATH+ 'groundtruth/'
    print("[INFO]: new MASK_DATA_PATH : "+ MASK_DATA_PATH)
    print("[INFO]: new IMAGE_DATA_PATH : " + IMAGE_DATA_PATH)
    IMAGES_FILENAMES = os.listdir(IMAGE_DATA_PATH)
    print("[INFO]: There are " + str(len(IMAGES_FILENAMES)) + " found")

np.random.seed = SEED
print("[INFO]: Loading images into RAM", flush = True)
X = np.zeros((len(IMAGES_FILENAMES), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((len(IMAGES_FILENAMES), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for n, filename in tqdm(enumerate(IMAGES_FILENAMES), total=len(IMAGES_FILENAMES)):
    img = imread(IMAGE_DATA_PATH + filename)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X[n] = img
    mask = imread(MASK_DATA_PATH + filename)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
    if USE_GENERATED_IMG:
        Y[n] = mask[:,:,0]
    else:
        Y[n] = mask

x_train=X
y_train=Y


