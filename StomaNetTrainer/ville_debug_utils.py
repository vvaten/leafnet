import os
from PIL import Image
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import cv2

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vvaten/rsync/leafnet/leafnet/leafnet_libs')
sys.path.insert(1, '/home/vvaten/rsync/leafnet/leafnet/leafnet_libs/stoma_detector')

print(f"PATH={sys.path}")

from load_image import load_image
from stoma import sample_loader, generate_heatmap

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


class PredictAfterEachTrainingEpoch(tf.compat.v1.keras.callbacks.Callback):
    def __init__(self, predict_model, predict_image_name, model_input_shape, model_output_shape, image_denoiser, resize_ratio):
        self.predict_model = predict_model
        self.predict_preview_path = "training_preview"
        if not os.path.exists(self.predict_preview_path):
            os.makedirs(self.predict_preview_path)
        self.background_type = True
        if image_denoiser:
            preprocessor = image_denoiser.denoise
        else:
            preprocessor = None
        self.predict_image, _ = load_image(predict_image_name, resize_ratio, self.background_type, preprocesser = preprocessor)
        print(f"Predicting a preview after each epch with image {predict_image_name}")
        filename = f"{self.predict_preview_path}/predict_epoch__source.png"
        Image.fromarray(self.predict_image).convert('RGB').save(filename)
        self.model_input_shape = model_input_shape
        self.model_output_shape = model_output_shape

    def on_epoch_end(self, epoch, logs=None):
        result_array = generate_heatmap(self.predict_image, self.predict_model, self.model_input_shape, self.model_output_shape)
        filename = f"{self.predict_preview_path}/predict_epoch_{epoch:02d}.png"
        Image.fromarray(np.uint8(result_array*255)).convert('RGB').save(filename)



