#!/bin/bash

set -e

INTERACTION_DATA="./data/yeast/protein.actions.tsv"
SEQUENCE_DATA="./data/yeast/protein.dictionary.tsv"

mkdir -p "./Result/yeast"

SAVE_DIR="./Result/yeast/PPIGAN_paper_aligned_val_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SAVE_DIR/Log"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0 \
python -u ./run/train_ppigan_paper_aligned.py \
--cuda \
--interaction_data "$INTERACTION_DATA" \
--sequence_data "$SEQUENCE_DATA" \
--save_dir "$SAVE_DIR" \
--epoch 50 \
--batch_size 256 \
--d_steps 2 \
--g_steps 1 \
--d_lr 0.0001 \
--g_lr 0.0001 \
--beta_real_loss 1.0 \
--beta_fake_loss 0.005 \
--lambda_freq 30.0 \
--freq_warmup_epochs 10 \
--noise_scale 0.3 \
--val_ratio 0.2 \
--threshold 0.5 \
--seed 42 \
--save_interval 1 \
--log_interval 20 \
> "$LOG_DIR/train.log" 2>&1 &

echo "Started training"
echo "PID: $!"
echo "SAVE: $SAVE_DIR"
echo "LOG: $LOG_DIR/train.log"
