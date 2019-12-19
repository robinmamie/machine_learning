# Notebooks

The notebooks present in this directory where used for the development of the different models.

Please note that these jupyter notebooks where developped on Google Colab. If you are going to run these notebooks note that you will need to change the paths to the different files. Note that these notebooks are quite similar to each other but vary slightly from version to version.

Please find below a description of the different notebooks:

* `unet.ipynb` : U-Net implementation
* `unet++.ipynb` : UNet++ implementation
* `unet++_deep_BinCrossEnt.ipynb` : UNet++ with deep supervision. Loss function is binary cross entropy
* `unet++_deep_hybrid.ipynb` : UNet++ with deep supervision. Loss function is a combination of the binary cross entropy and the dice coeficients

These notebooks contain all the pipeline to load, train, tune hyper parameters, create F1 on validation set, and predict test images.

This folder also contains `baseline_cnn.py`, which is the modified CNN model that was given at the start of the project. It creates 800 images and uses them to train during 100 epochs.
