#!/bin/bash

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  train_2.py \
  --model DiT-S/8 \
  --data-path /mnt/shared_dir/scratch/tfm_luis/DiT/Data/Train \
  --num-conditioning-images 100 \
  --conditioning-path /mnt/shared_dir/scratch/tfm_luis/DiT/Data/Test \
  --results-dir ./results \
  --global-batch-size 32 \
  --checkpoint DiT-XL-2-256x256.pt \
  >> output_s8_32_small_2.txt 2>&1 &