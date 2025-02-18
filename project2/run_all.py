import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

import argparse
import datetime
import gc
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

def main(epochs):
    # RTX fix
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #allow_growth_session = tf.compat.v1.Session(config=config)
    #tf.compat.v1.keras.backend.set_session(allow_growth_session)
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    # NON TWEAKABLE CONSTANTS
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

    # TWEAKABLE
    # Image generation
    GENERATE_NEW_IMG   = False
    USE_GENERATED_IMG  = True
    IMG_TO_GEN_PER_IMG = 200

    # Load existing model
    USE_SAVED_MODEL = True
    TRAIN_MODEL     = True

    # Predictions
    RUN_PREDICTIONS_ON_TEST_IMAGES = False

    if GENERATE_NEW_IMG:
        # load the input image, convert it to a NumPy array, and then
        # reshape it to have an extra dimension
        for img in tqdm(IMAGES_FILENAMES):
            image = load_img(IMAGE_DATA_PATH+img)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            truth = load_img(MASK_DATA_PATH+img)
            truth = img_to_array(truth)
            truth = np.expand_dims(truth, axis=0)
            # construct the image generator for data augmentation then
            # initialize the total number of images generated thus far
            aug = ImageDataGenerator(rotation_range=360,
                    zoom_range=0.3,
                    brightness_range=[0.7,1],
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    vertical_flip=True,
                    shear_range=0.15,
                    horizontal_flip=True,
                    fill_mode="reflect")
            total = 0

            # construct the actual Python generator
            imageGen = aug.flow(image, y=truth, batch_size=1, save_to_dir=OUTPUT_DATA_IMAGE_PATH + "images",
            save_prefix=img.split(".")[0], save_format="png", seed = SEED )
            truthGen = aug.flow(truth, y=truth, batch_size=1, save_to_dir=OUTPUT_DATA_IMAGE_PATH + "groundtruth",
            save_prefix=img.split(".")[0], save_format="png", seed = SEED )
            # loop over examples from our image data augmentation generator
            for image in imageGen:
                # increment our counter
                total += 1

                # if we have reached the specified number of examples, break
                # from the loop
                if total == IMG_TO_GEN_PER_IMG:
                    break

            total = 0
            for image in truthGen:
                # increment our counter
                total += 1

                # if we have reached the specified number of examples, break
                # from the loop
                if total == IMG_TO_GEN_PER_IMG:
                    break

    if(USE_GENERATED_IMG):
        print("[INFO] Updating images_filename")
        IMAGE_DATA_PATH = OUTPUT_DATA_IMAGE_PATH+'images/'
        MASK_DATA_PATH = OUTPUT_DATA_IMAGE_PATH+ 'groundtruth/'
        print("[INFO] new MASK_DATA_PATH : "+ MASK_DATA_PATH)
        print("[INFO] new IMAGE_DATA_PATH : " + IMAGE_DATA_PATH)
        IMAGES_FILENAMES = os.listdir(IMAGE_DATA_PATH)
        print("[INFO] There are " + str(len(IMAGES_FILENAMES)) + " found")
    
    np.random.seed = SEED
    print("[INFO] Loading images into RAM", flush = True)
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

    # Build U-Net++ model
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

    # One DenseNet Node
    d_u1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    d1 = tf.keras.layers.concatenate([d_u1, c3])
    d1 = tf.keras.layers.Dense(64, activation='relu')(d1)

    u7 = tf.keras.layers.concatenate([u7, d1])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c7)

    d_u2 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    d2 = tf.keras.layers.concatenate([d_u2, c2])
    d2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(d2)

    d_u3 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d1)
    d3 = tf.keras.layers.concatenate([d_u3, d2, c2])
    d3 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(d3)


    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, d3])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c8)


    d_uc2 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
    d4 = tf.keras.layers.concatenate([d_uc2, c1])
    d4 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(d4)

    d_ud2 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d2)
    d5 = tf.keras.layers.concatenate([d_ud2, d4, c1])
    d5 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(d5)

    d_ud3 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d3)
    d6 = tf.keras.layers.concatenate([d_ud3, d5, c1])
    d6 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(d6)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, d6], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, kernel_initializer='he_normal',
                                padding='same')(c9)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    if USE_SAVED_MODEL:
        if not os.path.isfile(MODEL_SAVE_LOCATION):
            print("[ERROR] Could not locate file for model weights. Proceding without loading weights.")
        else:
            model.load_weights(MODEL_SAVE_LOCATION)
            print("[INFO] Loading saved model weights")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1)
    callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tensorboard_callback,
        cp_callback 
    ]

    gc.collect()
    if TRAIN_MODEL:
        print("[INFO] Training model")
        #total_epochs = epochs
        #while epochs > 0:
        model.fit(X, Y, validation_split=0.1, batch_size=4, epochs=epochs, callbacks=callbacks, shuffle=True)
        model.save_weights(MODEL_SAVE_LOCATION, overwrite=True)
        #    total_epochs += epochs
        #    epochs = int(input(" -> How many more epochs? "))
        #print(f"[INFO] Trained for {total_epochs} epochs.")

    gc.collect()

    def img_float_to_uint8(img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
        return rimg

    predictions = []
    if RUN_PREDICTIONS_ON_TEST_IMAGES:
        print("[INFO] Running prediction on submission set")
        if not os.path.isdir(PREDICTION_SUBMISSION_DIR):
            os.mkdir(PREDICTION_SUBMISSION_DIR)
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
    else:
        print("[INFO] Skipping predicting test images")

    best_threshold = 0.51
    if RUN_PREDICTIONS_ON_TEST_IMAGES:
        print("[INFO] Writing prediction to drive")

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
            Image.fromarray(cimg).save(PREDICTION_SUBMISSION_DIR + f"gt_{i}.png")
    else:
        print("[INFO] Skipping write of predictions to disk")

    # Creating ouput for submission
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

    # assign a label to a patch
    def patch_to_label(patch):
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(image_filename):
        """Reads a single image and outputs the strings that should go into the submission file"""
        img_number = int(re.search(r"\d+", image_filename).group(0))
        im = mpimg.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


    def masks_to_submission(submission_filename, *image_filenames):
        """Converts images into a submission file"""
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


    if RUN_PREDICTIONS_ON_TEST_IMAGES:
        print("[INFO] Parsing prediction for AICrowd")
        time = strftime("%Y%m%dT%H%M%S")
        submission_filename = f'submission-{time}.csv'
        image_filenames = []
        for i in range(1, 51):
            image_filename = f'{PREDICTION_SUBMISSION_DIR}gt_{i}.png'
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)
    else:
        print("[INFO] Skipping prediction for AICrowd")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide the number of epochs.')
    main(int(sys.argv[1]))
