import os
from PIL import Image
from tensorflow.keras import backend
import numpy as np
import cv2

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


def sample_loader(source_size, target_size, sample_img):
    # Get the shape of samples
    img_r_size, img_c_size = sample_img.shape

    r_max_step = math.ceil(img_r_size/target_size)
    c_max_step = math.ceil(img_c_size/target_size)

    network_border = (source_size-target_size)//2

    expand_r_size = (target_size*(r_max_step))+(network_border*2)
    expand_c_size = (target_size*(c_max_step))+(network_border*2)

    img_r_pad = expand_r_size-img_r_size-network_border
    img_c_pad = expand_c_size-img_c_size-network_border
    image_padded = cv2.copyMakeBorder(sample_img.astype(np.uint8), network_border,img_r_pad,network_border,img_c_pad,cv2.BORDER_REFLECT)

    # Sample_array, Sample_weight
    sample_array = np.zeros((r_max_step*c_max_step, source_size, source_size, 1),dtype=float)
    for r in range (r_max_step):
        for c in range (c_max_step):
            block_id = (r * c_max_step) + c
            start_r = r * target_size
            start_c = c * target_size
            sample_array[block_id] = image_padded[start_r:start_r+source_size,start_c:start_c+source_size,np.newaxis].astype(float) / 255
    return sample_array, r_max_step, c_max_step

def generate_heatmap(normalized_sample_image, model_object, model_input_shape, model_output_shape):
    # Get the shape of samples
    img_r_size, img_c_size = normalized_sample_image.shape
    # Input Image
    samples, r_max_step, c_max_step = sample_loader(model_input_shape, model_output_shape, normalized_sample_image)
    # Use the Model to Predict the Stomas
    results = model_object.predict(samples)[:,:,:,0]
    # Rearrange the Results to Get Heatmap of Stoma Distribution
    total_list = list()
    for r_step in range (r_max_step):
        c_list = list()
        for c_step in range (c_max_step):
            c_list.append(results[c_step+r_step*c_max_step,:,:])
        total_list.append(np.concatenate(c_list, axis = 1))
    result_array = np.concatenate(total_list, axis = 0)[:img_r_size,:img_c_size]

    return result_array



class PredictAfterEachTrainingEpoch(keras.callbacks.Callback):
    def __init__(self, predict_image_name, model_input_shape, model_output_shape, image_denoiser, resize_ratio)
        self.predict_model = self.model
        predict_preview_path = "training_preview"
        if not os.path.exists(predict_preview_path):
            os.makedirs(predict_preview_path)
        self.predict_image = load_image(predict_image_name, resize_ratio, background_type, preprocesser = image_denoiser)
        print(f"Predicting a preview after each epch with image {predict_image_name}")
        self.model_input_shape = model_input_shape
        self.model_output_shape = model_output_shape

    def on_train_epoch_end(self, epoch, logs=None):
        result_array = generate_heatmap(self.predict_image, self.predict_model, self.model_input_shape, self.model_output_shape)
        filename = f"{predict_preview_path}/predict_epoch_{epoch}.png"
        Image.fromarray(result_array).save(filename)
        print(f"Saved preview in {filename} after epoch {epoch}")

