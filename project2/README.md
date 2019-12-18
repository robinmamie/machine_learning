# Road Segmentation Project
#### Bastien Beuchat, Robin Mamie, Jeremy Mion

Image segmentation is a computer vision process in which images are partitioned into different segments. It has a key role to play in many different fields of research. Among its concrete applications are domains such as medical imaging, machine vision and in our case dynamic map creation. Acquiring aerial photography is a cheap and efficient way of collecting information about the topography of the terrain below. In this project, we set out to create a machine learning algorithm that detects the roads out of these photographs. Automatically detecting the location and width of roads is a very powerful tool that allows map-making companies to keep their data up to date with very little cost.

In this project, we explore the different possibilities by starting with a simple convolutional neural network, and ending up with one implementing UNet++ using deep supervision.

## Requirements

The file `requirements.txt` contains all required libraries to run our project.

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
                [--search-threshold] [--no-predict] [--no-aicrowd] [--rtx]

    Prediction runner for the EPFL ML Road Segmentation 2019 Challenge. The
    default behaviour loads our best model and creates its AICrowd submission.

    optional arguments:
    -h, --help           show this help message and exit
    -generate [number]   Number of images to generate per train set image
                        (default 0)
    --use-augmented-set  Use the generated augmented train set (default False)
    -model [number]      Number of the model to use (default 1)
    --no-load            Do not load any previously saved model (default True)
    -train [epochs]      Number of epochs to train the neural network with
                        (default 0)
    --search-threshold   Search the best threshold on a validation set (default
                        False)
    --no-predict         Do not predict any image from the test set (default
                        True)
    --no-aicrowd         Do not generate file for AICrowd submission (default
                        True)
    --rtx                Allow memory growth for RTX GPUs (default False)

### Model number

Here are the available models in the script run.py:

- 0: U-Net
- 1: U-Net++
- 2: U-Net++ with deep supervision
- 3: U-Net++ with deep supervision and a custom loss
