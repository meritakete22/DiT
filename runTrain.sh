#!/bin/bash

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  train.py \
  --model DiT-S/8 \
  --data-path /mnt/shared_dir/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train \
  --results-dir ./results \
  --global-batch-size 32 \
  >> results_s8_32_3.txt 2>&1 &