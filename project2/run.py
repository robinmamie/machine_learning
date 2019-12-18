import argparse as ap
import datetime
import gc
import matplotlib.image as mpimg
import numpy as np
import os
import random
import re
import tensorflow as tf

from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from time import strftime
from tqdm import tqdm

from smooth_tiled_predictions import predict_img_with_smooth_windowing

# Constants
## Seeding
SEED = 42

## Image definition
IMG_WIDTH    = 400
IMG_HEIGHT   = 400
IMG_CHANNELS = 3
PIXEL_DEPTH  = 255

## Folder definitions
IMAGE_DATA_PATH           = 'training/images/'
MASK_DATA_PATH            = 'training/groundtruth/'
MODEL_SAVE_LOCATION       = 'road_segmentation_model.h5'
SUBMISSION_DATA_DIR       = 'test_set_images/'
PREDICTION_SUBMISSION_DIR = 'predictions_submission/'
CHECKPOINT_PATH           = 'checkpoints/cp.ckpt'
IMAGES_FILENAMES          = os.listdir(IMAGE_DATA_PATH)
### Image generation
OUTPUT_DATA_IMAGE_PATH = 'augmented_set/'
VALIDATION_DATA_PATH   = 'validation_set/'

def parse_flags():
    """Parses the flags given at the execution."""
    parser = ap.ArgumentParser(description="""Prediction runner for the EPFL ML
        Road Segmentation 2019 Challenge. The default behaviour loads our best
        model and creates its AICrowd submission.""")
    parser.add_argument(
        '-generate',
        metavar='number',
        dest='generate',
        type=int,
        nargs='?',
        help='Number of images to generate per train set image (default 0)',
        default=0,
    )
    parser.add_argument(
        '--use-augmented-set',
        dest='augmented_set',
        action='store_true',
        help='Use the generated augmented train set (default False)',
    )
    parser.add_argument(
        '-model',
        metavar='number',
        dest='model',
        type=int,
        nargs='?',
        help='Number of the model to use (default 1)',
        default=1,
    )
    parser.add_argument(
        '--no-load',
        dest='load',
        action='store_false',
        help='Do not load any previously saved model (default True)',
    )
    parser.add_argument(
        '-train',
        metavar='epochs',
        dest='epochs',
        type=int,
        nargs='?',
        help='Number of epochs to train the neural network with (default 0)',
        default=0,
    )
    parser.add_argument(
        '--search-threshold',
        dest='threshold',
        action='store_true',
        help='Search the best threshold on a validation set (default False)',
    )
    parser.add_argument(
        '-min-threshold',
        metavar='limit',
        dest='min_threshold',
        type=float,
        nargs='?',
        help='Minimum threshold search (default 0.39)',
        default=0.39,
    )
    parser.add_argument(
        '-max-threshold',
        metavar='limit',
        dest='max_threshold',
        type=float,
        nargs='?',
        help='Maximum threshold search (default 0.41)',
        default=0.41,
    )
    parser.add_argument(
        '-step-threshold',
        metavar='step',
        dest='step_threshold',
        type=float,
        nargs='?',
        help='Step for the threshold search (default 1e-3)',
        default=1e-3,
    )
    parser.add_argument(
        '--no-predict',
        dest='predict',
        action='store_false',
        help='Do not predict any image from the test set (default True)',
    )
    parser.add_argument(
        '--no-aicrowd',
        dest='aicrowd',
        action='store_false',
        help='Do not generate file for AICrowd submission (default True)',
    )
    parser.add_argument(
        '--rtx',
        dest='rtx',
        action='store_true',
        help='Allow memory growth for RTX GPUs (default False)',
    )
    args = parser.parse_args()
    # TODO test incorrect (negative) values
    return args

def generate_images(number_to_generate, folder=OUTPUT_DATA_IMAGE_PATH):
    """Generates a new augmented training set.
    
    Parameters
    ----------
    nunmber_to_generate : int
        The number of images to generate per original training image
    folder : str
        The path to the set's folder
    """
    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension
    print(f"""[INFO] Generating {number_to_generate} images per training image
              in {folder}""")
    for img in tqdm(IMAGES_FILENAMES):
        image = load_img(IMAGE_DATA_PATH+img)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        truth = load_img(MASK_DATA_PATH+img)
        truth = img_to_array(truth)
        truth = np.expand_dims(truth, axis=0)
        # construct the image generator for data augmentation then
        # initialize the total number of images generated thus far
        aug = ImageDataGenerator(
            rotation_range=360,
            zoom_range=0.3,
            brightness_range=[0.7,1],
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=True,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="reflect"
        )
        # Define Python generators
        imageGen = aug.flow(
            image,
            y=truth,
            batch_size=1,
            save_to_dir=folder+"images",
            save_prefix=img.split(".")[0],
            save_format="png",
            seed = SEED
        )
        truthGen = aug.flow(
            truth,
            y=truth,
            batch_size=1,
            save_to_dir=folder+"groundtruth",
            save_prefix=img.split(".")[0],
            save_format="png",
            seed = SEED
        )
        total = 0
        # Loop over examples from our image data augmentation generator
        for image in imageGen:
            total += 1
            if total == number_to_generate:
                # If enough were generated, break out of the loop
                break
        total = 0
        for image in truthGen:
            total += 1
            if total == number_to_generate:
                break

def update_path_train_set(folder=OUTPUT_DATA_IMAGE_PATH):
    """Updates the global path pointing to the training set.
    
    Parameters
    ----------
    folder : str
        The path to the set's folder
    """
    print("[INFO] Updating images_filename")
    IMAGE_DATA_PATH = folder+'images/'
    MASK_DATA_PATH = folder+ 'groundtruth/'
    print("[INFO] new IMAGE_DATA_PATH : " + IMAGE_DATA_PATH)
    print("[INFO] new MASK_DATA_PATH : "+ MASK_DATA_PATH)
    IMAGES_FILENAMES = os.listdir(IMAGE_DATA_PATH)
    print("[INFO] There are " + str(len(IMAGES_FILENAMES)) + " found")

def build_unet_model(type):
    """Builds the corresponding U-Net model.
    
    Parameters
    ----------
    type : int
        The TensorFlow model type (0: U-Net, 1: UNet++, 2: UNet++ with
        deep supervision, 3: UNet++ with DS and custom loss)

    Returns
    -------
    tf.keras.Model
        The newly created TensorFlow model
    """

    def dice_coef(y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return ((2 * intersection + smooth)
            / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

    def bce_dice_loss(y_true, y_pred):
        return (tf.keras.losses.binary_crossentropy(y_true, y_pred)
            + dice_coef(y_true, y_pred))

    # Define U-Net input
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH,+ IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Down-sampling
    c1 = tf.keras.layers.Conv2D(
        16, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(
        16, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(
        32, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(
        32, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(
        64, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(
        64, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = tf.keras.layers.Conv2D(
        128, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(
        128, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = tf.keras.layers.Conv2D(
        256, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(
        256, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c5)

    # Up-sampling
    u6 = tf.keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(
        128, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(
        128, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c6)
    if type > 0:
        # Add dense layer #1
        d_u1 = tf.keras.layers.Conv2DTranspose(
            64, (2, 2), strides=(2, 2), padding='same')(c4)
        d1 = tf.keras.layers.concatenate([d_u1, c3])
        d1 = tf.keras.layers.Dense(
            64, activation=tf.keras.activations.relu)(d1)
        u7 = tf.keras.layers.concatenate([u7, d1])
    else:
        u7 = tf.keras.layers.concatenate([u7, c3])

    c7 = tf.keras.layers.Conv2D(
        64, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(
        64, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(c7)
    if type > 0:
        # Add dense layer #2
        d_u2 = tf.keras.layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding='same')(c3)
        d2 = tf.keras.layers.concatenate([d_u2, c2])
        d2 = tf.keras.layers.Dense(
            128, activation=tf.keras.activations.relu)(d2)

        d_u3 = tf.keras.layers.Conv2DTranspose(
            32, (2, 2), strides=(2, 2), padding='same')(d1)
        d3 = tf.keras.layers.concatenate([d_u3, d2, c2])
        d3 = tf.keras.layers.Dense(
            128, activation=tf.keras.activations.relu)(d3)

        u8 = tf.keras.layers.concatenate([u8, d3])
    else:
        u8 = tf.keras.layers.concatenate([u8, c2])

    c8 = tf.keras.layers.Conv2D(
        32, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(
        32, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding='same')(c8)
    if type > 0:
        # Add dense layer #3
        d_uc2 = tf.keras.layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding='same')(c2)
        d4 = tf.keras.layers.concatenate([d_uc2, c1])
        d4 = tf.keras.layers.Dense(
            256, activation=tf.keras.activations.relu)(d4)

        d_ud2 = tf.keras.layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding='same')(d2)
        d5 = tf.keras.layers.concatenate([d_ud2, d4, c1])
        d5 = tf.keras.layers.Dense(
            256, activation=tf.keras.activations.relu)(d5)

        d_ud3 = tf.keras.layers.Conv2DTranspose(
            16, (2, 2), strides=(2, 2), padding='same')(d3)
        d6 = tf.keras.layers.concatenate([d_ud3, d5, c1])
        d6 = tf.keras.layers.Dense(
            256, activation=tf.keras.activations.relu)(d6)
        u9 = tf.keras.layers.concatenate([u9, d6], axis=3)
    else:
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

    c9 = tf.keras.layers.Conv2D(
        16, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(
        16, (3, 3), activation=tf.keras.activations.relu,
        kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    if type > 1:
        # Add deep supervision
        output_d4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
        output_d5 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d5)
        output_d6 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d6)
        model = tf.keras.Model(
            inputs=[inputs],
            outputs=[outputs, output_d4, output_d5, output_d6]
        )
        # Add custom loss if asked
        loss = bce_dice_loss if type == 3 else 'binary_crossentropy'
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy']
        )
    else:
        # No deep supervision
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def load_model(model):
    """Loads a previously computed model's weights."""
    if os.path.isfile(MODEL_SAVE_LOCATION):
        print("[INFO] Loading saved model weights")
        model.load_weights(MODEL_SAVE_LOCATION)
    else:
        print("""[ERROR] Could not locate file for model weights. Proceding
            without loading weights.""")

def load_images(is_generated):
    """Loads the previously selected images into the RAM.
    
    Parameters
    ----------
    is_generated : bool
        Whether the training images are generated or not

    Returns
    -------
    ndarray
        The training set's features
    ndarray
        The training set's labels
    """
    print("[INFO] Loading images into RAM", flush = True)
    X = np.zeros((len(IMAGES_FILENAMES), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                 dtype=np.uint8)
    Y = np.zeros((len(IMAGES_FILENAMES), IMG_HEIGHT, IMG_WIDTH, 1),
                 dtype=np.bool)

    for n, filename in tqdm(enumerate(IMAGES_FILENAMES),
                            total=len(IMAGES_FILENAMES)):   
        img = imread(IMAGE_DATA_PATH + filename)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                     preserve_range=True)
        X[n] = img
        mask = imread(MASK_DATA_PATH + filename)
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH),
                                     mode='constant', preserve_range=True),
                              axis=-1)
        if is_generated:
            Y[n] = mask[:,:,0]
        else:
            Y[n] = mask
    return X, Y

def train(model, epochs, is_generated, type):
    """Trains the TensorFlow model.
    
    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow model used for the predictions
    epochs : int
        The number of epochs to train the model with
    is_generated : bool
        Whether the training images are generated or not
    type : int
        The TensorFlow model type (0: U-Net, 1: UNet++, 2: UNet++ with
        deep supervision, 3: UNet++ with DS and custom loss)
    """
    X, Y = load_images(is_generated)
        
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH, 
        save_weights_only=True,
        verbose=1
    )
    callbacks = [
        tensorboard_callback,
        cp_callback 
    ]
    gc.collect()

    print("[INFO] Training model")
    if type > 1:
        Yp = [Y,Y,Y,Y]
    else:
        Yp = Y
    model.fit(X, Yp, validation_split=0.1, batch_size=4, epochs=epochs,
              callbacks=callbacks, shuffle=True)
    model.save_weights(MODEL_SAVE_LOCATION, overwrite=True)
    gc.collect()

def compute_best_threshold(model, type, lower, upper, step):
    """Computes the best threshold used for predictions, according to
    the number of misclassified tiles on a generated validation set.
    
    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow model used for the predictions
    type : int
        The TensorFlow model type (0: U-Net, 1: UNet++, 2: UNet++ with
        deep supervision, 3: UNet++ with DS and custom loss)
    lower : float
        The search's lower bound
    upper : float
        The search's upper bound
    step : float
        The search's step size

    Returns
    -------
    float
        The best prediction threshold
    """
    # best foreground_threshold: missclasified tiles count
    NUMBERS_OF_IMAGES_TO_USE = 100 # Number of images to classify

    # Load images
    update_path_train_set(VALIDATION_DATA_PATH)
    X, Y = load_images(is_generated=False)

    # assign a label to a patch
    def patch_to_label(patch, fg):
        df = np.mean(patch)
        if df > fg:
            return 1
        else:
            return 0

    def mask_to_submission_strings(im, fg):
        patch_size = 16
        mask = np.zeros((im.shape[1]//patch_size, im.shape[0]//patch_size))
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                mask[i//patch_size, j//patch_size] = patch_to_label(patch, fg)
        return mask

    def get_prediction(img, fg):
        x=np.array(img)
        x=np.expand_dims(x, axis=0)
        predict = model.predict(x, verbose=0)
        if type > 2:
            predict = predict[0]
        predict = (predict - predict.min())/(predict.max() - predict.min())
        predict = np.squeeze(predict)
        return mask_to_submission_strings(predict, fg)

    print('[INFO] Computing the best prediction threshold')
    number_of_pixels_off = []  #average number of missclasified images
    fg_values =  np.arange(lower, upper+step, step)
    for idx, fg in tqdm(enumerate(fg_values), total=len(fg_values)):
        total = 0
        for idx in range(NUMBERS_OF_IMAGES_TO_USE):
            prediction = get_prediction(X[idx], fg)
            total += (np.abs(prediction
                - mask_to_submission_strings(np.squeeze(Y[idx]), fg)).sum())
        number_of_pixels_off.append(total / NUMBERS_OF_IMAGES_TO_USE)
    best_threshold = fg_values[np.argmin(number_of_pixels_off)]
    print(f'[INFO] Best foreground_threshold value : {best_threshold}')
    min = np.min(number_of_pixels_off)
    print(f'[INFO] Average number of missclasified tiles : {min}')
    return best_threshold

def predict(model, type):
    """Predicts the test set images using the given model and its type.
    
    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow model used for the predictions
    type : int
        The TensorFlow model type (0: U-Net, 1: UNet++, 2: UNet++ with
        deep supervision, 3: UNet++ with DS and custom loss)
    """
    def img_float_to_uint8(img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
        return rimg

    print("[INFO] Running prediction on submission set")
    predictions = []
    if not os.path.isdir(PREDICTION_SUBMISSION_DIR):
        os.mkdir(PREDICTION_SUBMISSION_DIR)
    for i in range(1, 51):
        pimg = imread(SUBMISSION_DATA_DIR + f"test_{i}.png")[:,:,:IMG_CHANNELS]
        predictions.append(predict_img_with_smooth_windowing(
            pimg,
            window_size=IMG_WIDTH,
            subdivisions=2,  # Minimal amount of overlap for windowing
            nb_classes=1,
                pred_func=(
                    lambda img_batch_subdiv:
                        model.predict(img_batch_subdiv) if type < 2
                        else model.predict(img_batch_subdiv)[0]
                )
            )
        )

    print("[INFO] Writing prediction to drive")
    pred = np.array(predictions.copy())
    for i in range(1, 51):
        pimg = pred[i-1]
        w = pimg.shape[0]
        h = pimg.shape[1]
        cimg = np.zeros((w, h, 3), dtype=np.uint8)
        pimg *= 255
        pimg = pimg.astype(np.uint8)
        pimg8 = np.squeeze(img_float_to_uint8(pimg))
        cimg[:, :, 0] = pimg8
        cimg[:, :, 1] = pimg8
        cimg[:, :, 2] = pimg8
        Image.fromarray(cimg).save(PREDICTION_SUBMISSION_DIR + f"gt_{i}.png")

def predict_aicrowd(foreground_threshold):
    """Creates a submission for AIcrowd.
    
    Parameters
    ----------
    foreground_threshold : float
        Ratio of pixels to be set for a patch to be predicted as a road
    """
    def patch_to_label(patch):
        """Assign a label to a patch"""
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(image_filename):
        """Reads a single image and outputs the strings that should go
        into the submission file
        """
        img_number = int(re.search(r"\d+", image_filename).group(0))
        im = mpimg.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


    def masks_to_submission(submission_filename, *image_filenames):
        """Converts images into a submission file
        """
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s)
                             for s in mask_to_submission_strings(fn))

    print("[INFO] Parsing prediction for AICrowd")
    time = strftime("%Y%m%dT%H%M%S")
    submission_filename = f'submission-{time}.csv' # TODO keep this for submission?
    image_filenames = []
    for i in range(1, 51):
        image_filename = f'{PREDICTION_SUBMISSION_DIR}gt_{i}.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

def main():
    """Main function of the script"""
    args = parse_flags()

    if args.rtx:
        # Force the graphical memory growth to True (for RTX cards)
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)

    if args.generate > 0:
        # Generate the augmented training set
        generate_images(args.generate)
    
    if (args.threshold
            and not [f for f in os.listdir(VALIDATION_DATA_PATH+'images')
                     if not f.startswith('.')]):
        # Genereate the augmented validation set for thresholding, if
        # necessary and if asked for
        generate_images(1, folder=VALIDATION_DATA_PATH)

    # Fix numpy seed *after* using the Keras image generator (otherwise
    # it crashes)
    np.random.seed = SEED

    if args.augmented_set:
        # Update the path if user asked to use the augmented set
        update_path_train_set()

    # Build the corresponding U-Net model
    model = build_unet_model(args.model)

    if args.load:
        # Load any previously computed weights
        load_model(model)

    if args.epochs > 0:
        # Train the model
        train(
            model,
            epochs=args.epochs,
            is_generated=args.augmented_set,
            type=args.model,
        )

    if args.threshold:
        # Find the best threshold for the submission creation, i.e. how
        # many pixels must be set for a patch to be predicted as a road
        best_threshold = compute_best_threshold(
            model,
            type=args.model,
            lower=args.min_threshold,
            upper=args.max_threshold,
            step=args.step_threshold,
        )
    else:
        # Use the default threshold, optimized for our best model
        best_threshold = 0.4

    if args.predict:
        # Make predictions on images and save them on the drive
        predict(model, args.model)
    else:
        print("[INFO] Skipping predicting test images")

    if args.aicrowd:
        # Create the AIcrowd submission using the best threshold
        predict_aicrowd(best_threshold)
    else:
        print("[INFO] Skipping prediction for AICrowd")


if __name__ == '__main__':
    # Create folders
    if not os.path.exists(OUTPUT_DATA_IMAGE_PATH):
        os.makedirs(OUTPUT_DATA_IMAGE_PATH)
    if not os.path.exists(VALIDATION_DATA_PATH):
        os.makedirs(VALIDATION_DATA_PATH)
    if not os.path.exists(VALIDATION_DATA_PATH+'images'):
        os.makedirs(VALIDATION_DATA_PATH+'images')
    if not os.path.exists(VALIDATION_DATA_PATH+'groundtruth'):
        os.makedirs(VALIDATION_DATA_PATH+'groundtruth')
    # Set relevant seeds
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed = SEED
    tf.random.set_seed(SEED)
    main()
