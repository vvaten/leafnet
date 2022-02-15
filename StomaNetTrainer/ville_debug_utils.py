import os
from PIL import Image
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import cv2

import sys
from pathlib import Path

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vvaten/rsync/leafnet/leafnet/leafnet_libs')
sys.path.insert(1, '/home/vvaten/rsync/leafnet/leafnet/leafnet_libs/stoma_detector')

print(f"PATH={sys.path}")

from load_image import load_image
from stoma import sample_loader, generate_heatmap

def np_mem(x):
	return f"Array size: {x.size}, Item size: {x.itemsize}, Bytes: {x.size*x.itemsize}"

def save_preprocessed_image_samples(input_training_samples, input_training_labels, folder):
    index_to_save = 1
    image_count = input_training_samples[index_to_save].shape[0]

    print(f"Saving processed training samples and labels (the first array), size={image_count}")
    print(f"input_training_samples[0].shape {input_training_samples[0].shape}")
    print(f"input_training_labels[0].shape {input_training_labels[0].shape}")

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(image_count):
        imagearray = input_training_samples[index_to_save][i,:,:,0]
        tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
        tmp_image.save(f"{folder}/training_sample_{i:03d}" + '.png')
        imagearray = input_training_labels[index_to_save][i,:,:,0]
        tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
        tmp_image.save(f"{folder}/training_label_a_{i:03d}" + '.png')
        if input_training_labels[index_to_save].shape[-1] > 1:
            imagearray = input_training_labels[index_to_save][i,:,:,1]
            tmp_image = Image.fromarray(np.uint8(imagearray*255)).convert('RGB') # L?
            tmp_image.save(f"{folder}/training_label_b_{i:03d}" + '.png')
        print(f"Wrote images index {i}")

def dice_coeff(y_true, y_pred, smooth=1):
  intersection = backend.sum(y_true * y_pred, axis=[1,2,3])
  union = backend.sum(y_true, axis=[1,2,3]) + backend.sum(y_pred, axis=[1,2,3])
  dice = backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


class PredictAfterEachTrainingEpoch(tf.compat.v1.keras.callbacks.Callback):
    def __init__(self, predict_model, predict_folder, predict_n_images, target_folder, model_input_shape, model_output_shape, image_denoiser, resize_ratio):
        self.predict_model = predict_model
        self.predict_preview_path = target_folder
        if not os.path.exists(self.predict_preview_path):
            os.makedirs(self.predict_preview_path)
        self.background_type = True
        if image_denoiser:
            preprocessor = image_denoiser.denoise
        else:
            preprocessor = None

        self.predict_images = list()

        self.image_file_names = os.listdir(predict_folder)[:predict_n_images]
        print(f"Predicting a preview after each epch with image(s) {self.image_file_names}")

        for file_name in self.image_file_names:
            tmp_image, _ = load_image(os.path.join(predict_folder, file_name), resize_ratio, self.background_type, preprocesser = preprocessor)
            filename = os.path.join(self.predict_preview_path, Path(file_name).stem + "__original.png")
            Image.fromarray(tmp_image).convert('RGB').save(filename)
            self.predict_images.append(tmp_image)

        self.model_input_shape = model_input_shape
        self.model_output_shape = model_output_shape

    def on_epoch_end(self, epoch, logs=None):
        for i, file_name in enumerate(self.image_file_names):
            result_array = generate_heatmap(self.predict_images[i], self.predict_model, self.model_input_shape, self.model_output_shape)
            filename = os.path.join(self.predict_preview_path, Path(file_name).stem + f"_epoch_{epoch:02d}.png")
            Image.fromarray(np.uint8(result_array*255)).convert('RGB').save(filename)



