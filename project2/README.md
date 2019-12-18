# Road Segmentation Project
#### Bastien Beuchat, Robin Mamie, Jeremy Mion

Image segmentation is a computer vision process in which images are partitioned into different segments. It has a key role to play in many different fields of research. Among its concrete applications are domains such as medical imaging, machine vision and in our case dynamic map creation. Acquiring aerial photography is a cheap and efficient way of collecting information about the topography of the terrain below. In this project, we set out to create a machine learning algorithm that detects the roads out of these photographs. Automatically detecting the location and width of roads is a very powerful tool that allows map-making companies to keep their data up to date with very little cost.

In this project, we explore the different possibilities by starting with a simple convolutional neural network, and ending up with one implementing UNet++ using deep supervision.

## Requirements

The script was developped using `Python 3.7.5`. The file `requirements.txt` contains all required libraries to run our project. To install all requirements, use the following command, using pip:

    pip install -r requirements.txt

Here is an explanatory list for our requirements (and their version number):

- tensorflow (2.0.0): model creation and training
- Keras (2.2.4): model creation and training
- numpy (1.17.4): general purpose library
- matplotlib (3.1.1): helper for the AIcrowd submission
- tqdm (4.40.2): shows user friendly progress bars in the console
- scikit_image (0.15.0): image handling
- Pillow (6.2.1): image handling

### Other external libraries

We also make use of the local file `smooth_tiled_predictions.py`. It was fetched from [this repository](https://github.com/Vooban/Smoothly-Blend-Image-Patches) (commit 2f5866bce03ac5edfecd1bacfdd8a0663c659f09), created by Guillaume Chevalier.
This file is very useful to apply our neural network on test images that are bigger than our training images.

## Folder structure

    .
    ├── archive                      # Contains different notebooks used for development
    ├── augmented_set                # Created with the provided script
    │   ├── groundtruth
    │   └── images
    ├── checkpoints                  # Created to save training checkpoints
    ├── predictions_submission       # Folder where the predictions are saved
    ├── report
    │   └── ...                      # Report files
    ├── smooth_tiled_predictions.py  # External library, predict using bigger images than training set
    ├── test_set_images              # Folder created by unzipping test_set_images.zip and putting all the images at its root
    │   ├── test_1.png
    │   ├── ...
    │   └── test_50.png
    ├── training                     # Folder created by unzipping training.zip
    │   ├── groundtruth
    │   └── images
    ├── README.md
    ├── requirements.txt
    ├── run.py                       # The main script
    └── validation_set               # Created for dynamic thresholding
        ├── groundtruth
        └── images

## Create the best submission

Simply run the following:

    python run.py

To create our best submission from scratch, please use the following:

    python run.py -generate 100 --use-augmented-set --no-load -train 100 --search-threshold

## Flags

The help message of the script shows this:

    usage: run.py [-h] [-generate [number]] [--use-augmented-set]
                [-model [number]] [--no-load] [-train [epochs]]
                [--search-threshold] [-min-threshold [limit]]
                [-max-threshold [limit]] [-step-threshold [step]] [--no-predict]
                [--no-aicrowd] [--rtx]

    Prediction runner for the EPFL ML Road Segmentation 2019 Challenge. The
    default behaviour loads our best model and creates its AICrowd submission.

    optional arguments:
    -h, --help            show this help message and exit
    -generate [number]    Number of images to generate per train set image
                            (default 0)
    --use-augmented-set   Use the generated augmented train set (default False)
    -model [number]       Number of the model to use (default 1)
    --no-load             Do not load any previously saved model (default True)
    -train [epochs]       Number of epochs to train the neural network with
                            (default 0)
    --search-threshold    Search the best threshold on a validation set (default
                            False)
    -min-threshold [limit]
                            Minimum threshold search (default 0.39)
    -max-threshold [limit]
                            Maximum threshold search (default 0.41)
    -step-threshold [step]
                            Step for the threshold search (default 1e-3)
    --no-predict          Do not predict any image from the test set (default
                            True)
    --no-aicrowd          Do not generate file for AICrowd submission (default
                            True)
    --rtx                 Allow memory growth for RTX GPUs (default False)


### Model number

Here are the available models in the script run.py:

- 0: U-Net
- 1: U-Net++
- 2: U-Net++ with deep supervision
- 3: U-Net++ with deep supervision and a custom loss
