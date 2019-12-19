# Notebooks

The notebooks present in this direcotry where used for the development of the different models.

Please note that these jupyter notebooks where developped on Google Colab. If you are going to run these notebooks locally please note that you will need to change the paths to the different files. Note that these notebooks are quite similar ot each other but vary slightly from version to version.

* **unet.ipynb** : U-Net implementation
* **unet++.ipynb** : UNet++ implementation
* **unet++_deep_BinCrossEnt.ipynb** : UNet++ with deep supervision. Loss function is binary cross entropy
* **unet++_deep_hybrid.ipynb** : UNet++ with deep supervision. Loss function is a combination of the binary cross entropy and the dice coeficients

These notebooks contain all the pipeline to load, train, tune hyper parameters, and predict the images.