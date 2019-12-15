## Tries

### 1. 07.12.2019 UNet

```
SEED = 42
IMG_TO_GEN_PER_IMG = 100
aug = ImageDataGenerator(rotation_range=360,
                  zoom_range=0.3,
                  brightness_range=[0.7,1],
                  width_shift_range=0.1,
                  height_shift_range=0.1,
                  vertical_flip=True,
                  shear_range=0.15,
                  horizontal_flip=True,
                  fill_mode="reflect")
```

i.e. 10'000 images, split between 9'000 for training and 1'000 for validation

400 epochs, weights and loss/accuracy evolution saved on Robin's home computer
batch size = 32
validation_split = 0.1

Prediction thresholds (not foreground thresholds):
- 0.100 -> 0.863 F1 AICrowd
- 0.250 -> 0.881 F1 AICrowd
- 0.514 -> 0.893 F1 AICrowd


## UNet++

1500 images generated with ImageDataGenerator (get specs from my inital commit on github, beofre robin changed stuff)

15 epochs batch size 1
10 epochs batch size 4 (wanting a less noisy gradient)

Deciding best threshold with F1 settings:
### F1-score estimation
NUMBER_OF_IMG_TO_TEST = 20

best_threshold = 0.52

AIcowd = 0.872 F1 -> Secondary 0.931


_____
15 epochs batch size 1
170 epochs batch size 4 (wanting a less noisy gradient)

AIcowd = 0.875 F1	-> 0.933 Secondary



## UNet++
```
50 epochs
batch size 1
10'000 images
split 0.1
best_threshold 0.506
Training time required on our PC: 8h20
```
____
 F1 = 0.888, secondary 0.937

______
Same model but without best threshold. Using AI_crowd submision creation with parameter 
 
foreground_threshold = 0.25. 

Feeding the output probability masks to create submission yields F1 = 0.884 Secondary = 0.934
____ 
foreground_threshold = 0.28

F1 = 0.891 Secondary =	0.939
_____

foreground_threshold = 0.31

F1 = 0.897 Secondary =	0.943

____

foreground_threshold = 0.33

F1 = 0.901	Secondary = 0.945

____

foreground_threshold = 0.40

***F1 = 0.905 Secondary = 0.949***


___

Threshold to the accuracy of 0.001

best foreground_threshold value : 0.378

Given best threshold average number of missclasified tiles : 11.5

F1: 0.904	Secondary: 0.948

___
Threshold to the accuracy of 0.001

best foreground_threshold value : 0.347

Given best threshold average number of missclasified tiles : 11.52

F1 : 0.902	Secondary 0.946


## UNet++ DeepSupervision

With Robin's run with to following settings:

```
50 epochs
batch size 1
10'000 images
split 0.1
Training time required on our PC: 9h30
```

```
# best foreground_threshold:
NUMBERS_OF_IMAGES_TO_USE = 50
MIN_FOREGROUND_VALUE = 0.25
MAX_FOREGOURND_VALUE = 0.45
STEP = 0.01
 ```
 
 Best threshold is **0.40**
 
 F1 score -> 0.888 Secondary -> 0.941

____

## UNet (same as try 1)
No thresholding on the prediction image generation.
Using thresholding on submission generation F1 -> 0.904

**Threshold to the accuracy of 0.01**

Best threshold is **0.37**

Using thresholding on submission generation F1 -> 0.904

**Threshold to the accuracy of 0.001**


best foreground_threshold value : 0.358
Given best threshold average number of missclasified tiles : 9.96

F1 : 0.905	Secondary :0.950
____
## UNet++ DeepSupervision with diceLoss
```
 Batch size 1
 10'000 images
 50 epochs
 split 0.1
 Training time required on our PC: 9h30
```

```
# best foreground_threshold:
NUMBERS_OF_IMAGES_TO_USE = 50
MIN_FOREGROUND_VALUE = 0.30
MAX_FOREGOURND_VALUE = 0.70
STEP = 0.01
```

The contrast of the predicted images was verry low so we applied a linear scalling to the values to get them between 0 and 1.

**Threshold to the accuracy of 0.01**

F1 0.819, secondary : 0.901

best foreground_threshold value : 0.5400000000000003

Given best threshold average number of missclasified tiles : 62.64

**Threshold to the accuracy of 0.001**
best foreground_threshold value : 0.535
Given best threshold average number of missclasified tiles : 58.29

F1 0.820	Secondary 0.901
