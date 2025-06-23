#!/bin/bash

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  train_2.py \
  --model DiT-S/8 \
  --data-path /mnt/shared_dir/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train \
  --num-conditioning-images 100 \
  --conditioning-path /mnt/shared_dir/datasets/ImageNet/ILSVRC/Data/CLS-LOC/test \
  --results-dir ./results \
  --global-batch-size 32 \
  --num-epochs 10 \
  >> results_s8_32_3.txt 2>&1 &