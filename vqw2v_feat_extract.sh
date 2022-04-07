#!/bin/bash

conda_env=torch_1.7
source activate $conda_env


audio_dir=vcc20
python  vqwv2vec_feat_extract.py vqw2v/vq-wav2vec_kmeans.pt $audio_dir dump/vqw2v_feat_test/
