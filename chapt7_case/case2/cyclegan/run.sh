#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python train.py --n_epochs 800 --batchSize 64 --lr 1e-4 --show_freq 10 --lambda1 1.0 --lambda2 10.0 --n_cpu 8


python test.py