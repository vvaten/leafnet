#!/bin/bash

# Run a short train & eval cycle for validating if the TF2 porting is still working

export TF_ENABLE_DEPRECATION_WARNINGS=1

python train_model.py train -i /workspace/anne_test_data/stanford/sample-dic/ -l /workspace/anne_test_data/stanford/label/ -o /workspace/self_trained/tf2_port_test3 --sample_res 8.0 --validation_split 0.2 --optimizer Nadam --eval_image_dir /workspace/anne_test_data/stanford_test/sample-dic/ --eval_label_dir /workspace/anne_test_data/stanford_test/label/ --predict_preview 4 --batch_size 100 -e 200 --gaussian_blur 0 --duplicate_rotate --duplicate_mirror --duplicate_rotate_stomata_only 30 --duplicate_rescale_stomata_only 10,10 --shuffle_training\
&& \
python train_model.py eval -o /workspace/self_trained/tf2_port_test2_eval -m /workspace/self_trained/tf2_port_test3.model --sample_res 8.0 --eval_image_dir /workspace/anne_test_data/stanford_test/sample-dic/ --eval_label_dir /workspace/anne_test_data/stanford_test/label/ --gaussian_blur 0 --predict_preview 4
