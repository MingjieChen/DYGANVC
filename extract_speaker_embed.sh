#!/bin/bash


conda_env=torch_1.7
source activate $conda_env

python extract_speaker_embed.py \
       vcc20/ \
       dump/ppg-vc-spks \
       speaker_encoder/ckpt/pretrained_bak_5805000.pt \
       speaker.json \


