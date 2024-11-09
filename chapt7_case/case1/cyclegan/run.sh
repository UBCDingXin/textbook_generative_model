#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --n_epochs 400 --decay_epoch 200 --batchSize 128 --lr 1e-4 --lambda1 1.0 --lambda2 10.0
