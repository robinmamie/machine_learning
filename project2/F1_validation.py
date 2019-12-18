# best foreground_threshold: f1 score

'''
Please have a functional model definned
'''
THRESHOLD_VALUE_TO_TEST = 0.40

from sklearn.metrics import f1_score


def get_F1_threshold(x_val, y_val, threshold_value):
  '''
  let x_val be an array of length [100,400,400,3]. These are the images from the validation set
  let y_val be an array of shape [100,400,400,1]. These are the masks of the ground truth
  let threshold_value be the value of the threshold to test
  '''

  assert x_val.shape[0] >= 100 , "x_val is not large enough. First dimmension less than 100 needs to be greater than 100"
  assert y_val.shape[0] >= 100, "y_val is not large enough. First dimmension less than 100 needs to be greater than 100"
  
  NUMBERS_OF_IMAGES_TO_USE = 100      # This is the number of images to use durring the calculation.

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
    
  accumulator = 0
  for idx in tqdm(range(NUMBERS_OF_IMAGES_TO_USE),desc='Calculating F1 score'):
    prediction = get_prediction(x_val[idx], threshold_value)
    accumulator += f1_score(prediction.astype(np.bool).flatten(), mask_to_submission_strings(np.squeeze(y_val[idx]), threshold_value).astype(np.bool).flatten(), average='binary')
  print( f'\nf1 socre is : { accumulator / NUMBERS_OF_IMAGES_TO_USE}')

#get_F1_threshold(x_train, y_train, 0.4)
