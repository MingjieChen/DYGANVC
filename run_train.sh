#!/bin/bash

conda_env=torch_1.7
source activate $conda_env




python train.py --config_path config.yml
                   
