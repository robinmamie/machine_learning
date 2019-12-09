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
SUBMISSION_DATA_DIR       = 'test_set_images/'
PREDICTION_SUBMISSION_DIR = 'predictions_submission/'

IMAGES_FILENAMES = os.listdir(IMAGE_DATA_PATH)

# Image generation
OUTPUT_DATA_IMAGE_PATH = 'augmented_set/'

# Seeding
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed = SEED
tf.random.set_seed(SEED)

# Checkpoints
checkpoint_path = 'checkpoints/cp.ckpt'
checkpoint_dir  = os.path.dirname(checkpoint_path)

# Image generation
GENERATE_NEW_IMG   = False
USE_GENERATED_IMG  = True
IMG_TO_GEN_PER_IMG = 100

# Load existing model
USE_SAVED_MODEL = False

# F1-score estimation
NUMBER_OF_IMG_TO_TEST = 10

# Predictions
RUN_PREDICTIONS_ON_TEST_IMAGES = True

# Create models
models = []
for i in range(10):
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH,+ IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    models.append(model)

# Load models
if USE_SAVED_MODEL:
    for m, model in enumerate(models):
        path = f'road_segmentation_model_{m}.h5'
        if not os.path.isfile(path):
            print("[ERROR]: Could not locate file for model weights. Proceding without loading weights.")
        else:
            model.load_weights(path)
            print("[INFO]: Loading saved model weights")

# Run predictions
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

if RUN_PREDICTIONS_ON_TEST_IMAGES:
    print("[INFO]: Running prediction on submission set")
    if not os.path.isdir(PREDICTION_SUBMISSION_DIR):
        os.mkdir(PREDICTION_SUBMISSION_DIR)

    for m, model in enumerate(models):
        predictions = []
        for i in range(1, 51):
            pimg = imread(SUBMISSION_DATA_DIR + f"test_{i}.png")[:,:,:IMG_CHANNELS]
            predictions.append(predict_img_with_smooth_windowing(
                pimg,
                window_size=IMG_WIDTH,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=1,
                pred_func=(
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)
                )
                )
            )
        best_threshold = 0.5 # Default
        if RUN_PREDICTIONS_ON_TEST_IMAGES:
            print("[INFO]: Writing prediction to drive")

            ROAD_THRESHOLD = 0.25#0.1

            pred = np.array(predictions.copy())
            for i in range(1, 51):
                pimg = pred[i-1]
                w = pimg.shape[0]
                h = pimg.shape[1]
                cimg = np.zeros((w, h, 3), dtype=np.uint8)
                pimg = (pimg > best_threshold).astype(np.uint8)
                pimg8 = np.squeeze(img_float_to_uint8(pimg))
                cimg[:, :, 0] = pimg8
                cimg[:, :, 1] = pimg8
                cimg[:, :, 2] = pimg8
                Image.fromarray(cimg).save(PREDICTION_SUBMISSION_DIR + f"gt_{m}_{i}.png")
        else:
            print("[INFO]: Skipping write of predictions to disk")
else:
    print("[INFO]: Skipping predicting test images")

print("[INFO] End reached")