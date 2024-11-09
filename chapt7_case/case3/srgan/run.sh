#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --upscale_factor 4 --num_epochs 20 --train_batch_size 8 --test_batch_size 1 --num_works 8 --save_freq 5

