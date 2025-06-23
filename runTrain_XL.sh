#!/bin/bash

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  train_2.py \
  --model DiT-XL/2 \
  --data-path /mnt/shared_dir/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train \
  --num-conditioning-images 100 \
  --conditioning-path /mnt/shared_dir/datasets/ImageNet/ILSVRC/Data/CLS-LOC/test \
  --results-dir ./results \
  --global-batch-size 32 \
  --num-epochs 10 \
  --checkpoint DiT-XL-2-256x256.pt \
  >> output_xl_256 2>&1 &