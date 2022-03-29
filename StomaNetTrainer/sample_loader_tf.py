import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import multiprocessing

sample_loader_debug = False

eps = 1e-7

# Tensorflow version of the sample loader
def sample_loader_tf(source_size, target_size, sample_img, label_img, border_cutoff, stoma_weight, stomata_only):
    if sample_loader_debug: print(f"sample_loader(source_size={source_size}, target_size={target_size}, border_cutoff={border_cutoff}, stoma_weight={stoma_weight}")

    # Get the shape of samples
    image_shape = tf.shape(sample_img)
    img_r_size = image_shape[0]
    img_c_size = image_shape[1]
    
    if sample_loader_debug: print(f"sample_img.shape: {sample_img.shape}")
    r_max_step = tf.cast(tf.math.floor((img_r_size-(border_cutoff*2))/target_size), tf.int32)
    c_max_step = tf.cast(tf.math.floor((img_c_size-(border_cutoff*2))/target_size), tf.int32)

    if sample_loader_debug: print(f"r_max_step: {r_max_step}, c_max_step: {c_max_step}")

    network_border = (source_size-target_size)//2

    if sample_loader_debug: print(f"network_border: {network_border}")

    img_left_up_pad = network_border - border_cutoff

    if sample_loader_debug: print(f"img_left_up_pad: {img_left_up_pad}")

    # Cut/Pad Left and Top
    label_img = label_img[border_cutoff:,border_cutoff:]
    if img_left_up_pad > 0:
        sample_img = tf.pad(sample_img, [[img_left_up_pad,0],[img_left_up_pad,0], [0,0]],"REFLECT")
    elif img_left_up_pad < 0:
        sample_img = sample_img[-img_left_up_pad:,-img_left_up_pad:]

    # Cut/Pad Right and Bottom
    image_shape = tf.shape(sample_img)
    img_r_size = image_shape[0]
    img_c_size = image_shape[1]

    if sample_loader_debug: print(f"After cut/pad left/top sample_img.shape: {sample_img.shape}")
        
    label_img = label_img[:target_size*(r_max_step),:target_size*(c_max_step)]

    sample_target_r_len = target_size*(r_max_step) + source_size - target_size

    if sample_loader_debug: print(f"sample_target_r_len: {sample_target_r_len}")

    if sample_target_r_len > img_r_size:
        sample_img = tf.pad(sample_img, [[0,sample_target_r_len - img_r_size],[0,0],[0,0]],"REFLECT")
    elif sample_target_r_len < img_r_size:
        sample_img = sample_img[:sample_target_r_len,:]
    
    sample_target_c_len = target_size*(c_max_step) + source_size - target_size

    if sample_loader_debug: print(f"sample_target_c_len: {sample_target_c_len}")

    if sample_target_c_len > img_c_size:
        sample_img = tf.pad(sample_img, [[0,0],[0,sample_target_c_len - img_c_size],[0,0]],"REFLECT")
    elif sample_target_c_len < img_c_size:
        sample_img = sample_img[:,:sample_target_c_len]

    if sample_loader_debug: print(f"After cut/pad right/bottom sample_img.shape: {sample_img.shape}")

    # Normalize
    sample_img_norm = sample_img
    label_img_norm = label_img

    # Sample_array, Sample_weight
    sample_array = tf.TensorArray(tf.float32, size=r_max_step*c_max_step)
    label_array = tf.TensorArray(tf.float32, size=r_max_step*c_max_step)
    
    ### TODO: Try with single output only. No negative label image.

    multiply_area_list = list()

    for r in range (r_max_step):
        for c in range (c_max_step):
            block_id = (r * c_max_step) + c
            start_r = r * target_size
            start_c = c * target_size
            label_array = label_array.write(block_id, label_img_norm[start_r:start_r+target_size,start_c:start_c+target_size])
            sample_array = sample_array.write(block_id, sample_img_norm[start_r:start_r+source_size,start_c:start_c+source_size])
                                                                
    label_stack = label_array.stack()
    sample_stack = sample_array.stack()

    def stomata_only_func():
        contains_stoma_mask = tf.math.reduce_mean(label_stack,axis=[1,2,3]) > 0.1
        return tf.boolean_mask(sample_stack, contains_stoma_mask), tf.boolean_mask(label_stack, contains_stoma_mask)
    
    def no_filtering():
        return sample_stack, label_stack

    sample_stack, label_stack = tf.cond(stomata_only, stomata_only_func, no_filtering)

    def multiply_stoma_weight():
        # get a mask for stoma only samples
        contains_stoma_mask = tf.math.reduce_mean(label_stack,axis=[1,2,3]) > 0.1
        # get only those samples that contain a stoma
        label_stack_stomata_only = tf.boolean_mask(label_stack, contains_stoma_mask)
        sample_stack_stomata_only = tf.boolean_mask(sample_stack, contains_stoma_mask)
        # multiply those samples by the number of stoma_weight
        tiler = tf.concat([tf.reshape(stoma_weight, [1]), tf.constant([1,1,1], tf.int32)], axis=0)
        label_stack_stomata_tiled = tf.tile(label_stack_stomata_only, tiler)
        sample_stack_stomata_tiled = tf.tile(sample_stack_stomata_only, tiler)
        # concatenate them on top of the original unfiltered stacks
        return tf.concat([sample_stack, sample_stack_stomata_tiled], axis=0), tf.concat([label_stack, label_stack_stomata_tiled], axis=0)
    
    sample_stack, label_stack = tf.cond(tf.greater(stoma_weight, tf.constant(1)),
                                        multiply_stoma_weight,
                                        no_filtering)
    
    
    # add the negative target to the label stack
    label_stack = tf.concat([label_stack,1-label_stack], axis=3)
    
    return sample_stack, label_stack #, tf.shape(sample_stack)[0]
    
    
def read_image_and_label(full_path_filename, label_dir, resize_ratio, gaussian_blur):
    parts = tf.strings.split(full_path_filename, os.sep)
    filename_only = parts[-1]
    
    image = tf.io.read_file(full_path_filename)
    ###print(full_path_filename, image)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image_resize_target = [tf.cast(tf.multiply(tf.cast(tf.shape(image)[0], tf.float64), resize_ratio), tf.int32),
                           tf.cast(tf.multiply(tf.cast(tf.shape(image)[1], tf.float64), resize_ratio), tf.int32)]
    image = tf.image.resize(image, image_resize_target)
    
    
    label_full_path_filename = tf.strings.join([label_dir, filename_only], separator=os.sep)
    label = tf.io.read_file(label_full_path_filename)
    label = tf.io.decode_png(label)
    label = tf.image.convert_image_dtype(label, tf.float32)
    label = label[:,:,2:3]
    label = tf.image.resize(label, image_resize_target)
    if gaussian_blur > 0:
        label = tfa.image.gaussian_filter2d(label, filter_shape=[5*gaussian_blur+1,5*gaussian_blur+1], sigma=gaussian_blur)
    
    return image, label, False

def read_image_and_label_iterator(full_path_filename):
    return read_image_and_label(full_path_filename, label_dir, gaussian_blur)

def get_dataset_partitions(ds, ds_size, validation_split=0.2, shuffle=True):
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(ds_size, seed=12, reshuffle_each_iteration=False)
    
    val_size = int(validation_split * ds_size)
    train_size = ds_size - val_size
    
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size).take(train_size)
    
    return train_ds, val_ds

def add_mirrored_images(dataset):
    def mirror_images(image, label, stomata_only_transformed):
        return image[:,::-1], label[:,::-1], stomata_only_transformed
    
    return dataset.concatenate(dataset.map(mirror_images, num_parallel_calls=multiprocessing.cpu_count()))


def add_rot90_images(dataset):
    def rotate_90_degrees(image, label, stomata_only_transformed, k):
        return tf.image.rot90(image, k=k), tf.image.rot90(label, k=k), stomata_only_transformed

    return dataset\
            .concatenate(dataset.map(lambda x, y, z: rotate_90_degrees(x, y, z, 1), num_parallel_calls=multiprocessing.cpu_count()))\
            .concatenate(dataset.map(lambda x, y, z: rotate_90_degrees(x, y, z, 2), num_parallel_calls=multiprocessing.cpu_count()))\
            .concatenate(dataset.map(lambda x, y, z: rotate_90_degrees(x, y, z, 3), num_parallel_calls=multiprocessing.cpu_count()))
                       

def add_free_rotate_images(dataset, args_duplicate_rotate_stomata_only):

        
    def rotate_n_degrees(image, label, stomata_only_transformed, angle):
        # returns the image and label rotated and an integer telling that the image was rotated (1)
        return tfa.image.rotate(image, tf.constant(np.pi*angle/180.0)), tfa.image.rotate(label, tf.constant(np.pi*angle/180.0)), tf.constant(True)
    
    def nop(image, label, stomata_only_transformed):
        # returns the image and label directly and an integer telling that the image was not rotated (0)
        return image, label, stomata_only_transformed
    
    dataset_to_return = dataset.map(nop)
    
    if args_duplicate_rotate_stomata_only > 0:
        duplicate_rotate_stomata_only = range(0, 90, args_duplicate_rotate_stomata_only)
    else:
        return dataset_to_return
    
    for angle in duplicate_rotate_stomata_only[1:]:
        dataset_to_return = dataset_to_return\
            .concatenate(dataset.map(lambda x, y, z: rotate_n_degrees(x, y, z, angle), num_parallel_calls=multiprocessing.cpu_count()))
    
    return dataset_to_return


    
def add_zoomed_images(dataset, args_duplicate_rescale_stomata_only):
        
    def image_zoom(image, label, rescale_ops):
        width = tf.shape(image)[-3]
        height = tf.shape(image)[-2]
        scaled_width = tf.cast(tf.cast(width, tf.float32) * rescale_ops, tf.int32)
        scaled_height = tf.cast(tf.cast(height, tf.float32) * rescale_ops, tf.int32)

        rescaled_image = tf.image.resize(image, [scaled_width, scaled_height],
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=False,
                    antialias=True)

        original_size_image = tf.image.resize_with_crop_or_pad(rescaled_image, width, height)
        
        rescaled_label = tf.image.resize(label, [scaled_width, scaled_height],
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=False,
                    antialias=True)

        original_size_label = tf.image.resize_with_crop_or_pad(rescaled_label, width, height)

        return original_size_image, original_size_label, True

    def nop(image, label, transformed):
        # returns the image and label directly and an integer telling that the image was not manipulated
        return image, label, transformed
    
    dataset_to_return = dataset.map(nop)

    if len(args_duplicate_rescale_stomata_only) > 0 and ',' in args_duplicate_rescale_stomata_only:
        rescale_max = int(args_duplicate_rescale_stomata_only.split(',')[0])
        rescale_step = int(args_duplicate_rescale_stomata_only.split(',')[1])
        
        duplicate_rescale_stomata_only = np.arange(1.0-float(rescale_max)*0.01, 1.0+float(rescale_max)*0.01, float(rescale_step)*0.01)
    else:
        return dataset_to_return

    for rescale_ops in duplicate_rescale_stomata_only:
        if rescale_ops != 1.0:
            dataset_to_return = dataset_to_return\
                .concatenate(dataset.map(lambda x, y, z: image_zoom(x, y, rescale_ops), num_parallel_calls=multiprocessing.cpu_count()))
            
    return dataset_to_return

def load_sample_from_folder_tf(image_dir, label_dir, source_size, target_size, validation_split, image_denoiser, foreign_neg_dir=None, args_duplicate_undenoise=False, args_duplicate_invert=False, args_duplicate_mirror=False, args_duplicate_rotate=False, args_duplicate_rotate_stomata_only=0, args_duplicate_rescale_stomata_only="", resize_ratio = 1.0, stoma_weight = 1, gaussian_blur = 4):
    
    def map_to_imagestacks(image, label, stomata_only_transformed):
        return sample_loader_tf(source_size, target_size, image, label, 50, stoma_weight, stomata_only_transformed)

    if foreign_neg_dir is not None:
        raise ValueError("Foreign negative image data loading not implemented yet")
    
    if args_duplicate_undenoise:
        raise ValueError("undenoise not implemented yet")

    if args_duplicate_invert:
        raise ValueError("invert not implemented yet")
        
    if len(args_duplicate_rescale_stomata_only) > 0 and ',' in args_duplicate_rescale_stomata_only:
        rescale_max = int(args_duplicate_rescale_stomata_only.split(',')[0])
        rescale_step = int(args_duplicate_rescale_stomata_only.split(',')[1])
        
        duplicate_rescale_stomata_only = np.arange(1.0-float(rescale_max)*0.01, 1.0+float(rescale_max)*0.01, float(rescale_step)*0.01)
    else:
        duplicate_rescale_stomata_only = [1.0]

    print("Loading files from", str(image_dir + '*.png'))
    image_filenames_ds = tf.data.Dataset.list_files(str(image_dir + '*.png'), shuffle=False) #shuffle false, we'll shuffle them later.
    print("All dataset filenames:")
    for i in image_filenames_ds:
        print(i)

    train_dataset_filenames, validation_dataset_filenames = get_dataset_partitions(image_filenames_ds, len(image_filenames_ds), validation_split, True)
    
    print("Training dataset filenames:")
    for i in train_dataset_filenames:
        print(i)
        
    print("Validation dataset filenames:")
    for i in validation_dataset_filenames:
        print(i)


    train_dataset = train_dataset_filenames.map(lambda x: read_image_and_label(x,
                                                                         label_dir,
                                                                         resize_ratio,
                                                                         gaussian_blur),
                                          num_parallel_calls=multiprocessing.cpu_count())

    validation_dataset = validation_dataset_filenames.map(lambda x: read_image_and_label(x,
                                                                         label_dir,
                                                                         resize_ratio,
                                                                         gaussian_blur),
                                          num_parallel_calls=multiprocessing.cpu_count())

    if args_duplicate_mirror:
        train_dataset = add_mirrored_images(train_dataset)
        validation_dataset = add_mirrored_images(validation_dataset)
    
    if args_duplicate_rotate:
        train_dataset = add_rot90_images(train_dataset)
        validation_dataset = add_rot90_images(validation_dataset)

    if args_duplicate_rotate_stomata_only > 0:
        train_dataset = add_free_rotate_images(train_dataset, args_duplicate_rotate_stomata_only)

    if args_duplicate_rescale_stomata_only:
        train_dataset = add_zoomed_images(train_dataset, args_duplicate_rescale_stomata_only)
    
    train_dataset = train_dataset.map(map_to_imagestacks, num_parallel_calls=multiprocessing.cpu_count())
    validation_dataset = validation_dataset.map(map_to_imagestacks, num_parallel_calls=multiprocessing.cpu_count())
    
    return train_dataset, validation_dataset, len(train_dataset), len(validation_dataset), 0
