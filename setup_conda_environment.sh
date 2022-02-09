#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda update conda
conda update --all

conda create --name TF1.15 python=3.8
conda activate TF1.15

pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]

conda install -c conda-forge openmpi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/TF1.15/lib/

conda install Pillow==7.0
conda install matplotlib==3.1.3
conda install scikit-image==0.16.2
python -m pip install opencv-python==4.1.2.30
conda install scikit-learn==0.22.1

export XLA_FLAGS=--xla_hlo_profile
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
