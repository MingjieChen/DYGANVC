#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

[ ! -e vcc20 ] && echo "pleas download data first using './data_download.sh  vcc20'" && exit 1

audio_dir=vcc20
root=$PWD
[ ! -e dump ] && mkdir dump
python  vqwv2vec_feat_extract.py $audio_dir dump/vqw2v_feat_test/
