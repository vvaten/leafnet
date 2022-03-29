import argparse
import math

arg_parser = argparse.ArgumentParser(description="StomaNet model trainer.")
subparsers = arg_parser.add_subparsers(title='subcommands',
                                   description='valid subcommands',
                                   help='Action to perform: train or eval (Default: train)')


def set_train_mode():
    return True, False

def set_eval_mode():
    return False, True

arg_parser_train = subparsers.add_parser("train")

## Input
arg_parser_train.add_argument("-i", dest="image_dir", type=str, required=True, help="The folder containing training image.")
arg_parser_train.add_argument("-l", dest="label_dir", type=str, required=True, help="The folder containing training annotation.")
arg_parser_train.add_argument("-o", dest="save_path", type=str, required=True, help="The path to save trained model and/or results.")
arg_parser_train.add_argument("--eval_image_dir", dest="eval_image_dir", type=str, required=False, help="The folder containing evaluation image.")
arg_parser_train.add_argument("--eval_label_dir", dest="eval_label_dir", type=str, required=False, help="The folder containing evaluation annotation.")

arg_parser_train.add_argument("--sample_res", dest="sample_res", type=float, required=True, help="The resolution (px/μm) for training samples.")
arg_parser_train.add_argument("--target_res", dest="target_res", default=2.17, type=float, help="The target resolution (px/μm) of the trained model. Training samples will be automatically resized to match this resolution.(default value: 2.17)")

arg_parser_train.add_argument("-e", dest="epoch", type=int, default=10, help="The count of epochs,(a epoch means training the model on all training data once) (default value: 10)")
arg_parser_train.add_argument("-f", dest="foreign_neg_dir", type=str, help="(optional)The folder containing foreign negative images (images without stomata), foreign negative images will NOT be resized according to resolution.")
arg_parser_train.add_argument("-m", dest="stoma_net_model_path", type=str, help="The source model for transfer training or eval (must be another model based on StomaNet).")

# disabled while TF2 porting in progress
arg_parser_train.add_argument("--gpu_count", dest="multi_gpu", type=int, default=1, help="Tensorflow multi_gpu_model argument, use 1 for cpu or one gpu.(default value: 1)")
arg_parser_train.add_argument("--optimizer", dest="optimizer", type=str, default="SGD", help="Optimizer: SGD (with Nesterov momentum) or Nadam")
arg_parser_train.add_argument("--loss_function", dest="loss_function", type=str, default="kld", help="Loss function: kld or binary_crossentropy")
arg_parser_train.add_argument("--batch_size", dest="batch_size", type=int, default=40, help="Batch size of training.(default value: 40)")
arg_parser_train.add_argument("--validation_split", dest="validation_split", type=float, default=0.2, help="Images will be split into training and validation with this ratio BEFORE sample duplication.(default value: 0.2)")
arg_parser_train.add_argument("--dynamic_batch_size", dest="dynamic_batch_size", action="store_true", help="Seperate epochs to 5 training period with increasing batch sizes.")
arg_parser_train.add_argument("--shuffle_training", dest="shuffle_training", action="store_true", help="Shuffle training data")

arg_parser_train.add_argument("--duplicate_undenoise", dest="duplicate_undenoise", action="store_true", help="Duplicate samples(*2) by using denoised and raw images together.")
arg_parser_train.add_argument("--duplicate_invert", dest="duplicate_invert", action="store_true", help="Duplicate samples(*2) by using inverting images.")
arg_parser_train.add_argument("--duplicate_mirror", dest="duplicate_mirror", action="store_true", help="Duplicate samples(*2) by using mirrored images.")
arg_parser_train.add_argument("--duplicate_rotate", dest="duplicate_rotate", action="store_true", help="Duplicate samples(*4) by rotating samples 90, 180, 270 degree.")
arg_parser_train.add_argument("--duplicate_rotate_stomata_only", dest="duplicate_rotate_stomata_only", type=int, default=0, help="Duplicate stomata samples by rotating them in n degrees steps")
arg_parser_train.add_argument("--duplicate_rescale_stomata_only", dest="duplicate_rescale_stomata_only", type=str, default="", help="Duplicate stomata samples by rescaling them in +-n percent, with p percent steps, for example '10,5' creates -10%, -5%, +5%, +10% extra samples. '' for none")

arg_parser_train.add_argument("--predict_preview", dest="predict_preview", type=int, default=0, help="Record training progress by predicting with n images after each epoch")

arg_parser_train.add_argument("--stoma_weight", dest="stoma_weight", type=int, default=1, help="Image areas containing stomata will be mulitplied by this factor.(default value: 1)")
arg_parser_train.add_argument("--gaussian_blur", dest="gaussian_blur", type=int, default=4, help="Factor for Gaussian blur. (default: 4)")
arg_parser_train.set_defaults(set_train_or_eval_mode=set_train_mode)


arg_parser_eval = subparsers.add_parser("eval")
arg_parser_eval.add_argument("--eval_image_dir", dest="eval_image_dir", type=str, required=True, help="The folder containing evaluation image.")
arg_parser_eval.add_argument("--eval_label_dir", dest="eval_label_dir", type=str, required=True, help="The folder containing evaluation annotation.")
arg_parser_eval.add_argument("-o", dest="save_path", type=str, required=True, help="The path to save trained model and/or results.")
arg_parser_eval.add_argument("--sample_res", dest="sample_res", type=float, required=True, help="The resolution (px/μm) for training samples.")
arg_parser_eval.add_argument("-m", dest="stoma_net_model_path", type=str, required=True, help="The source model for evaluation.")
arg_parser_eval.add_argument("--batch_size", dest="batch_size", type=int, default=40, help="Batch size of training.(default value: 40)")
arg_parser_eval.add_argument("--gpu_count", dest="multi_gpu", type=int, default=1, help="Tensorflow multi_gpu_model argument, use 1 for cpu or one gpu.(default value: 1)")
arg_parser_eval.add_argument("--stoma_weight", dest="stoma_weight", type=int, default=1, help="Image areas containing stomata will be mulitplied by this factor.(default value: 1)")
arg_parser_eval.add_argument("--gaussian_blur", dest="gaussian_blur", type=int, default=4, help="Factor for Gaussian blur. (default: 4)")
arg_parser_eval.add_argument("--predict_preview", dest="predict_preview", type=int, default=0, help="Predict evaluatio n images")
arg_parser_eval.set_defaults(set_train_or_eval_mode=set_eval_mode)



import numpy as np
import os
import sys
import random

import tensorflow as tf

print("Tensorflow version", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
    
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, backend
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from gegl_denoise.denoiser import denoiser
from sample_loader import load_sample_from_folder
from sample_loader_tf import load_sample_from_folder_tf
from stoma_net_models import build_stoma_net_model

from ville_debug_utils import *

def shuffle_along_first_axis(samples, labels):
    assert samples.shape[0] == labels.shape[0]
    idx = tf.convert_to_tensor(np.random.rand(samples.shape[0]).argsort(axis=0))
    new_samples = tf.gather(samples, idx, axis=0)
    del samples
    new_labels = tf.gather(labels, idx, axis=0)
    del labels
    return new_samples, new_labels

## Parse Args
args = arg_parser.parse_args()
train_mode, eval_mode = args.set_train_or_eval_mode()

if train_mode:
    if args.epoch <= 0:
        raise ValueError(f"Epoch must be an intenger above zero!")

    image_dir = args.image_dir
    label_dir = args.label_dir
    save_path = args.save_path
    if not save_path.endswith(".model"): save_path += ".model"

    if args.eval_image_dir is not None and args.eval_label_dir is not None:
        # We have evaluation images as well. Perform evaluation after training.
        eval_mode = True

    sample_res = args.sample_res
    target_res = args.target_res

    total_epoch = args.epoch
    final_batch_size = args.batch_size
    dynamic_batch_size = args.dynamic_batch_size
    shuffle_training = args.shuffle_training
    foreign_neg_dir = args.foreign_neg_dir
    stoma_net_model_path = args.stoma_net_model_path
    multi_gpu = max(1, args.multi_gpu)
    loss_function = args.loss_function
    
    predict_preview = args.predict_preview

    optm = None
    if args.optimizer=="SGD":
        print("Using SGD optimizer")
        optm = optimizers.SGD(lr=0.001, momentum=0.9, nesterov = True)
    elif args.optimizer=="Nadam":
        print("Using Nadam optimizer")
        optm = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    else:
        raise ValueError("Unknown optimizer, allowed values are SGD and Nadam")

    validation_split = args.validation_split

    if final_batch_size <=0:
        raise ValueError("Batch size must be above zero.")

    if validation_split <= 0 or validation_split >= 1:
        raise ValueError("Validation split must be above zero and under one.")

    args_duplicate_undenoise=args.duplicate_undenoise
    args_duplicate_invert=args.duplicate_invert
    args_duplicate_mirror=args.duplicate_mirror
    args_duplicate_rotate=args.duplicate_rotate
    args_duplicate_rotate_stomata_only = args.duplicate_rotate_stomata_only
    args_duplicate_rescale_stomata_only = args.duplicate_rescale_stomata_only

    duplicate_times = 1
    if args_duplicate_undenoise: duplicate_times *= 2
    if args_duplicate_invert: duplicate_times *= 2
    if args_duplicate_mirror: duplicate_times *= 2
    if args_duplicate_rotate: duplicate_times *= 4
    if args_duplicate_rotate_stomata_only > 0: duplicate_times *= (90 // args_duplicate_rotate_stomata_only)
    if len(args_duplicate_rescale_stomata_only) > 0:
        rescale_max = int(args_duplicate_rescale_stomata_only.split(',')[0])
        rescale_step = int(args_duplicate_rescale_stomata_only.split(',')[1])
        duplicate_times *= (2*rescale_max/rescale_step+1)

    stoma_weight = args.stoma_weight
    if stoma_weight < 1:
        raise ValueError("Stoma weight must be an intenger above zero.")

    gaussian_blur = args.gaussian_blur

if eval_mode:
    eval_image_dir = args.eval_image_dir
    eval_label_dir = args.eval_label_dir
    stoma_net_model_path = args.stoma_net_model_path
    sample_res = args.sample_res
    if not train_mode:
        save_path = args.save_path
        if not save_path.endswith(".model"): save_path += ".model"

        # we need to parse these args as they were not parsed in the train_mode parser above
        loss_function = 'kld' # the model compile requires a loss function
        optm = optimizers.SGD(lr=0.001, momentum=0.9, nesterov = True) # the model compile requires a optimizer function
        multi_gpu = max(1, args.multi_gpu)
        stoma_weight = args.stoma_weight
        if stoma_weight < 1:
            raise ValueError("Stoma weight must be an intenger above zero.")
        gaussian_blur = args.gaussian_blur
        final_batch_size = args.batch_size
        predict_preview = args.predict_preview


print(f"Args:\n{str(vars(args))}")
print(f"Train: {train_mode}, Eval: {eval_mode}")

#Ignore tensorflow import warnings
# import warnings
# warnings.filterwarnings("ignore")

np.random.seed(0)
random.seed(0)
tf.compat.v1.set_random_seed(0)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
tf.compat.v1.keras.backend.set_session(sess)

training_sample_array = None
training_label_array = None
validation_sample_array = None
validation_label_array = None
training_dataset = None
validation_dataset = None

if multi_gpu > 1:
    strategy = tf.distribute.MirroredStrategy()
else:  # Use the Default Strategy
    strategy = tf.distribute.get_strategy()

if train_mode:
    print("Training.")
    # Compile model
    print("Compiling model...", end="")
    sys.stdout.flush()
    
    with strategy.scope():
        training_model = build_stoma_net_model(small_model=True, sigmoid_before_output=True)
    print(f"training_model shapes input.shape: {training_model.input.shape}, output.shape: {training_model.output.shape}")
    source_size = int(training_model.input.shape[1])
    target_size = int(training_model.output.shape[1])
    print("Done!")

    # Load Weight if used
    if stoma_net_model_path:
        print("Loading weights...", end="")
        sys.stdout.flush()
        # Not using load_weights to avoid tensorflow model issue 2676
        with strategy.scope():
            reference_model = load_model(stoma_net_model_path)
            training_model.set_weights(reference_model.get_weights())
        print("Done!")

    # Load Samples
    image_denoiser = None #denoiser()

    exec_model = training_model
    exec_model.compile(loss=loss_function, optimizer=optm, metrics=['accuracy', dice_coeff])

    callbacks = []

    if predict_preview:
        training_preview_predictor_callback = PredictAfterEachTrainingEpoch(exec_model, eval_image_dir, eval_label_dir, predict_preview, save_path[:-6]+"_training_preview", source_size, target_size, image_denoiser, target_res/sample_res)
        callbacks.append(training_preview_predictor_callback)

    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.75,
                                  patience=5,
                                  mode="min",
                                  min_lr=0.0001,
                                  cooldown=5,
                                  verbose=1)
    callbacks.append(reduce_lr)

    model_checkpoint_filepath = save_path[:-6]+'.checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_filepath,
        save_weights_only=True,
        monitor='val_dice_coeff',
        mode='max',
        save_best_only=True,
        verbose=1)
    callbacks.append(model_checkpoint_callback)
    
    early_stopping = EarlyStopping(
        monitor="val_dice_coeff",
        min_delta=0.001,
        patience=15,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    
    print("Loading samples...", end="")
    sys.stdout.flush()
    #input_training_samples, input_training_labels, input_validation_samples, input_validation_labels, img_count_sum, validation_count_sum, foreign_count_sum = load_sample_from_folder(image_dir, label_dir, source_size, target_size, validation_split, image_denoiser, foreign_neg_dir, args_duplicate_undenoise, args_duplicate_invert, args_duplicate_mirror, args_duplicate_rotate, args_duplicate_rotate_stomata_only, args_duplicate_rescale_stomata_only, target_res/sample_res, stoma_weight, gaussian_blur)
    
    training_dataset, validation_dataset, img_count_sum, validation_count_sum, foreign_count_sum = load_sample_from_folder_tf(image_dir, label_dir, source_size, target_size, validation_split, image_denoiser, foreign_neg_dir, args_duplicate_undenoise, args_duplicate_invert, args_duplicate_mirror, args_duplicate_rotate, args_duplicate_rotate_stomata_only, args_duplicate_rescale_stomata_only, target_res/sample_res, stoma_weight, gaussian_blur)
    print("Done!")

    print("Collected "+str(img_count_sum)+" sample images("+str(img_count_sum-validation_count_sum)+" for training, "+str(validation_count_sum)+" for validation) and "+str(foreign_count_sum)+" foreign negative images.")
    print("Input images are duplicated by *" + str(duplicate_times))

    ###save_preprocessed_image_samples(input_training_samples, input_training_labels, "tmp_stanford-dic_input_data_sample")
    print("original len training datase", tf.data.experimental.cardinality(training_dataset).numpy())

    #training_dataset = training_dataset.unbatch()
    #print("unbatched len training datase", tf.data.experimental.cardinality(training_dataset).numpy())
    
    #training_dataset = training_dataset.shuffle(5000).batch(final_batch_size).cache().repeat()

    def configure_for_performance(ds):
      ds = ds.unbatch()
      ds = ds.cache()
      if shuffle_training:
        ds = ds.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
      ds = ds.batch(final_batch_size)
      ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
      return ds
      
    training_dataset = configure_for_performance(training_dataset)
    validation_dataset = configure_for_performance(validation_dataset)

    #validation_dataset = validation_dataset.unbatch()
    #print("unbatched len validation_dataset", len(validation_dataset))
    #validation_dataset = validation_dataset.shuffle(5000).batch(final_batch_size).cache().repeat()

    #if shuffle_training:
    #    print("Shuffling training dataset")
     #   training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)


    #training_dataset = training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    #validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    #training_sample_array = tf.convert_to_tensor(np.concatenate(input_training_samples, axis=0))
    #del input_training_samples
    #training_label_array = tf.convert_to_tensor(np.concatenate(input_training_labels, axis=0))
    #del input_training_labels
    #validation_sample_array = tf.convert_to_tensor(np.concatenate(input_validation_samples, axis=0))
    #del input_validation_samples
    #validation_label_array = tf.convert_to_tensor(np.concatenate(input_validation_labels, axis=0))
    #del input_validation_labels

    #val_data = (validation_sample_array, validation_label_array)

    histories = []
    
    #batch_sizes = np.ones(total_epoch, dtype=int)*final_batch_size

    history = exec_model.fit(training_dataset,
                             epochs = total_epoch,
                             validation_data = validation_dataset,
                             #steps_per_epoch=100,
                             #validation_steps=10,
                             callbacks=callbacks)
    histories.append(history)
        
    print(f"Training complete. Saving model to {save_path}")

    saving_model = build_stoma_net_model(small_model=False, sigmoid_before_output=False)
    #saving_model.set_weights(training_model.get_weights())
    saving_model.load_weights(model_checkpoint_filepath)
    saving_model.save(save_path)
    print("Model saved!")
    with open(save_path[:-6]+".res", "w") as model_meta: model_meta.write(str(target_res))
    print(f"Wrote .res: {target_res}")
    with open(save_path[:-6]+".args", "w") as args_file: args_file.write(str(vars(args)))
    print(f"Wrote .args: {vars(args)}")
    history = {}
    for recorded_history in histories:
        for prop in recorded_history.history:
            if prop not in history:
                history[prop] = recorded_history.history[prop]
            else:
                history[prop].extend(recorded_history.history[prop])
    with open(save_path[:-6]+".training_history.json", "w") as history_file: history_file.write(str(history))
    print(f"Wrote .training_history.json: {str(history)}")


if eval_mode:
    print("Evaluation.")

    if training_sample_array is not None: del training_sample_array
    if training_label_array is not None: del training_label_array
    if validation_sample_array is not None: del validation_sample_array
    if validation_label_array is not None: del validation_label_array
    if training_dataset is not None: del training_dataset
    if validation_dataset is not None: del validation_dataset

    # Compile model
    print("Compiling model...", end="")
    sys.stdout.flush()
    with strategy.scope():
        evaluation_model = build_stoma_net_model(small_model=True, sigmoid_before_output=True)
    source_size = int(evaluation_model.input.shape[1])
    target_size = int(evaluation_model.output.shape[1])
    print("Done!")

    # Not using load_weights to avoid tensorflow model issue 2676
    if not train_mode:
        print(f"Loading weights from {stoma_net_model_path}")
        with strategy.scope():
            reference_model = load_model(stoma_net_model_path)
            evaluation_model.set_weights(reference_model.get_weights())
    else:
        print(f"Loading weights from trained model")
        with strategy.scope():
            evaluation_model.set_weights(training_model.get_weights())

    if 'target_res' not in globals():
        try:
            with open(stoma_net_model_path[:-6]+".res") as target_res_file:
                target_res = float(target_res_file.readline())
            print(f"Using target resolution {target_res} loaded from model location {stoma_net_model_path}.")
        except e:
            raise ValueError(f"Target resolution file not found from model file location! ({stoma_net_model_path})")

    image_denoiser = None #denoiser()

    exec_model = evaluation_model
    exec_model.compile(loss=loss_function, optimizer=optm, metrics=['accuracy', dice_coeff])

    print("Loading evaluation samples...", end="")
    sys.stdout.flush()
    eval_dataset, _, img_count_sum, validation_count_sum, foreign_count_sum = load_sample_from_folder_tf(eval_image_dir, eval_label_dir, source_size, target_size, 0, image_denoiser, None, False, False, False, False, False, "", target_res/sample_res, stoma_weight, args.gaussian_blur)
    print("Done!")

    eval_dataset = eval_dataset.unbatch().batch(final_batch_size)
    eval_dataset = eval_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    #evaluation_sample_array = np.concatenate(input_training_samples, axis=0)
    #del input_training_samples
    #evaluation_label_array = np.concatenate(input_training_labels, axis=0)
    #del input_training_labels

    #print(f"Mem evaluation_sample_array: {np_mem(evaluation_sample_array)}, shape: {evaluation_sample_array.shape}")
    #print(f"Mem evaluation_label_array: {np_mem(evaluation_label_array)}, shape: {evaluation_label_array.shape}")

    results = exec_model.evaluate(eval_dataset)#, batch_size=final_batch_size)

    if predict_preview:
        training_preview_predictor_callback = PredictAfterEachTrainingEpoch(exec_model, eval_image_dir, eval_label_dir, predict_preview, save_path[:-6]+"_training_preview", source_size, target_size, image_denoiser, target_res/sample_res)
        training_preview_predictor_callback.on_epoch_end(0)

    results_dict = dict(zip(['test_loss','test_acc','test_dice_coeff'], results))
    print(f"Results: {results_dict}")
    with open(save_path[:-6]+".results.json", "w") as results_file: results_file.write(str(results_dict))
    print(f"Wrote {save_path[:-6]}.results.json")


