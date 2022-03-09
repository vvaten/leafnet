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

# This Dice Coefficient only takes the positive label images into account, ignores the inversed negative label image
def dice_coeff2(y_true, y_pred, smooth=1):
    y_true = y_true[:,:,:,:1]
    y_pred = y_pred[:,:,:,:1]
    intersection = backend.sum(y_true * y_pred, axis=[1,2,3])
    union = backend.sum(y_true, axis=[1,2,3]) + backend.sum(y_pred, axis=[1,2,3])
    dice = backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


class PredictAfterEachTrainingEpoch(tf.compat.v1.keras.callbacks.Callback):
    def __init__(self, predict_model, predict_folder, predict_label_dir, predict_n_images, target_folder, model_input_shape, model_output_shape, image_denoiser, resize_ratio):
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
        self.predict_image_labels = list()

        self.image_file_names = os.listdir(predict_folder)[:predict_n_images]
        print(f"Predicting a preview after each epch with image(s) {self.image_file_names}")

        for file_name in self.image_file_names:
            tmp_image, _ = load_image(os.path.join(predict_folder, file_name), resize_ratio, self.background_type, preprocesser = preprocessor)
            filename = os.path.join(self.predict_preview_path, Path(file_name).stem + "__original.png")
            Image.fromarray(tmp_image).convert('RGB').save(filename)
            self.predict_images.append(tmp_image)
            label_PIL_image = Image.open(os.path.join(predict_label_dir, file_name)).convert('RGB')
            resized_label_PIL_image = label_PIL_image.resize((tmp_image.shape[1], tmp_image.shape[0]), Image.ANTIALIAS)
            #print(f"image shapes: label_PIL_image.size: {label_PIL_image.size}, resized_label_PIL_image.size: {resized_label_PIL_image.size}")

            tmp_label = np.array(resized_label_PIL_image)[:,:,2].astype(float) / 255
            self.predict_image_labels.append(tmp_label)

            #print(f"image shapes: tmp_image.shape {tmp_image.shape}, label_PIL_image.size: {label_PIL_image.size}, tmp_label.shape: {tmp_label.shape}")

        self.model_input_shape = model_input_shape
        self.model_output_shape = model_output_shape
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # track current epoch number internally as we are calling fit from scratch in every epoch
        self.current_epoch += 1
        for i, file_name in enumerate(self.image_file_names):
            result_array = generate_heatmap(self.predict_images[i], self.predict_model, self.model_input_shape, self.model_output_shape)
            filename = os.path.join(self.predict_preview_path, Path(file_name).stem + f"_epoch_{self.current_epoch:02d}.png")
            Image.fromarray(np.uint8(result_array*255)).convert('RGB').save(filename)
            error_image = generate_error_image(self.predict_image_labels[i], result_array)
            error_image_filename = os.path.join(self.predict_preview_path, Path(file_name).stem + f"_epoch_{self.current_epoch:02d}_error.png")
            Image.fromarray(np.uint8(error_image*255), "RGB").save(error_image_filename)



import numpy as np

def generate_error_image(y_true, y_pred):
    # Black = TN
    # Green = TP
    # Red = FP
    # Blue = FN

    ground_truth_positive = y_true.copy()
    ground_truth_positive[ground_truth_positive < 0.5] = 0.0
    #print("ground_truth_positive", ground_truth_positive)

    ground_truth_negative = 1.0 - y_true
    ground_truth_negative[ground_truth_negative <= 0.5] = 0.0
    #print("ground_truth_negative", ground_truth_negative)

    predicted_positive = y_pred.copy()
    predicted_positive[predicted_positive < 0.5] = 0.0
    #print("predicted_positive", predicted_positive)

    predicted_negative = 1.0 - y_pred
    predicted_negative[predicted_negative < 0.5] = 0.0
    #print("predicted_negative", predicted_negative)

    TP = (ground_truth_positive + predicted_positive)*0.5
    TP[TP <= 0.5] = 0.0
    #print("TP", TP)

    TN = (ground_truth_negative + predicted_negative)*0.5
    TN[TN <= 0.5] = 0.0
    #print("TN", TN)

    FP = predicted_positive - ground_truth_positive
    FP[FP <= 0.5] = 0.0
    #print("FP", FP)

    FN = predicted_negative - ground_truth_negative
    FN[FN <= 0.5] = 0.0
    #print("FN", FN)

    return np.stack([FP,TP,FN], axis=y_pred.ndim)


def test_error_image():
    y_true = np.array([[1.0, 1.0, 0.0, 0.0]])
    y_pred = np.array([[1.0, 0.0, 1.0, 0.0]])

    result = generate_error_image(y_true, y_pred)

    assert result.ndim == y_true.ndim + 1
    assert result.shape[-1] == 3

    for dim in range(len(y_true.shape)):
        assert y_true.shape[dim] == result.shape[dim]

    correct_answer = np.array([[[0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]])

    assert (result == correct_answer).all()

def test_error_image2():
    y_true = np.array([[0.6, 0.6, 0.4, 0.4]])
    y_pred = np.array([[0.6, 0.4, 0.6, 0.4]])

    result = generate_error_image(y_true, y_pred)

    assert result.ndim == y_true.ndim + 1
    assert result.shape[-1] == 3

    for dim in range(len(y_true.shape)-1):
        assert y_true.shape[dim] == result.shape[dim]

    correct_answer = np.array([[[0.0, 0.6, 0.0],
                                [0.0, 0.0, 0.6],
                                [0.6, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]])

    assert (result == correct_answer).all()

def test_error_image3():
    y_true = np.array([[0.5, 0.5, 0.499, 0.499]])
    y_pred = np.array([[1.0, 0.499, 0.6, 0.499]])

    result = generate_error_image(y_true, y_pred)

    assert result.ndim == y_true.ndim + 1
    assert result.shape[-1] == 3

    for dim in range(len(y_true.shape)-1):
        assert y_true.shape[dim] == result.shape[dim]

    correct_answer = np.array([[[0.0, 0.75, 0.0],
                                [0.0, 0.0, 0.501],
                                [0.6, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]])

    assert (result == correct_answer).all()

def test_error_image4():
    y_true = np.random.rand(10*3*3).reshape(10,3,3)
    y_pred = np.random.rand(10*3*3).reshape(10,3,3)

    result = generate_error_image(y_true, y_pred)

    assert result.ndim == y_true.ndim + 1
    assert result.shape[-1] == 3

    for dim in range(len(y_true.shape)-1):
        assert y_true.shape[dim] == result.shape[dim]



test_error_image()
test_error_image2()
test_error_image3()
test_error_image4()
