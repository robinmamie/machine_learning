# Road Segmentation Project

## Requirements

The file `requirements.txt` contains all required libraries to run our project.

## Folder structure

    .
    ├── augmented_set
    │   ├── groundtruth
    │   └── images
    ├── checkpoints                  # Created to save training checkpoints
    ├── Datasets
    ├── predictions_submission
    ├── report
    │   └── ...                      # Report files
    │   smooth_tiled_predictions.py  # External library, predict bigger images
    ├── test_set_images
    │   ├── test_1.png
    │   ├── ...
    │   └── test_50.png
    ├── training
    │   ├── groundtruth
    │   └── images
    ├── README.md
    ├── requirements.txt
    └── run.py

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

- 0: U-Net
- 1: U-Net++
- 2: U-Net++ with deep supervision
- 3: U-Net++ with deep supervision and a custom loss