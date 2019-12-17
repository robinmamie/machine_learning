# best foreground_threshold: missclasified tiles count
NUMBERS_OF_IMAGES_TO_USE = 100      # This is the number of images to use durring the calculation.
MIN_FOREGROUND_VALUE = 0.30 #included
MAX_FOREGOURND_VALUE = 0.38   #included
STEP = 0.001

assert MIN_FOREGROUND_VALUE < MAX_FOREGOURND_VALUE , f'MIN_FOREGROUND_VALUE : {MIN_FOREGROUND_VALUE} must be smaller than MAX_FOREGOURND_VALUE : {MAX_FOREGOURND_VALUE}'

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
    predict = model.predict(x, verbose=0)[0]
    #predict = (predict > best_threshold).astype(np.uint8)
    #predict = predict /255
    predict = (predict - predict.min())/(predict.max() - predict.min())
    predict = np.squeeze(predict)
    
    return mask_to_submission_strings(predict, fg)


number_of_pixels_off = []  #average number of missclasified images
fg_values =  np.arange(MIN_FOREGROUND_VALUE,MAX_FOREGOURND_VALUE+STEP,STEP)
for idx, fg in tqdm(enumerate(fg_values), total= len(fg_values)):
  total = 0
  for idx in range(NUMBERS_OF_IMAGES_TO_USE):
    prediction = get_prediction(x_train[idx], fg)
    total += np.abs(prediction - mask_to_submission_strings(np.squeeze(y_train[idx]), fg)).sum()
    #print(total)
  number_of_pixels_off.append( total / NUMBERS_OF_IMAGES_TO_USE)
plt.plot(fg_values, number_of_pixels_off);
print(flush=True)
print( f'best foreground_threshold value : {fg_values[np.argmin(number_of_pixels_off)]}')
print(f'Given best threshold average number of missclasified tiles : {np.min(number_of_pixels_off)}')

# creating data file of values used
temp = zip(fg_values, number_of_pixels_off)
with open('threshold_search_data.dat', 'w') as f:
  f.write(f'NUMBERS_OF_IMAGES_TO_USE = {NUMBERS_OF_IMAGES_TO_USE}      # This is the number of images to use durring the calculation.\n \
MIN_FOREGROUND_VALUE = {MIN_FOREGROUND_VALUE} #included \n\
MAX_FOREGOURND_VALUE = {MAX_FOREGOURND_VALUE}  #included \n\
STEP = {STEP}\n')
  for t in temp:
    f.write(str(t)+'\n')
print('[INFO] : saved foreground_threshold data in file.')
