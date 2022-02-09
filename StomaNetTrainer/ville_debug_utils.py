import os
from PIL import Image
from tensorflow.keras import backend
import numpy as np

def np_mem(x):
	return f"Array size: {x.size}, Item size: {x.itemsize}, Bytes: {x.size*x.itemsize}"


def save_preprocessed_image_samples(input_training_samples, input_training_labels, folder):
    image_count = input_training_samples[0].shape[0]

    print(f"Saving processed training samples and labels (the first array), size={image_count}")
    print(f"input_training_samples[0].shape {input_training_samples[0].shape}")
    print(f"input_training_labels[0].shape {input_training_labels[0].shape}")

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(image_count):
        imagearray = input_training_samples[0][i,:,:,0]
        tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
        tmp_image.save(f"{folder}/training_sample_{i:03d}" + '.png')
        imagearray = input_training_labels[0][i,:,:,0]
        tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
        tmp_image.save(f"{folder}/training_label_a_{i:03d}" + '.png')
        imagearray = input_training_labels[0][i,:,:,1]
        tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
        tmp_image.save(f"{folder}/training_label_b_{i:03d}" + '.png')
        print(f"Wrote images index {i}")


def dice_coeff(y_true, y_pred, smooth=1):
  intersection = backend.sum(y_true * y_pred, axis=[1,2,3])
  union = backend.sum(y_true, axis=[1,2,3]) + backend.sum(y_pred, axis=[1,2,3])
  dice = backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
