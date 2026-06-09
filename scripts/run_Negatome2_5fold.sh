#!/bin/bash

set -e

INTERACTION_DATA="./data/Negatome2/protein.actions.tsv"
SEQUENCE_DATA="./data/Negatome2/protein.dictionary.tsv"

mkdir -p "./fuse_Result/Negatome2_5fold"

SAVE_DIR="./fuse_Result/Negatome2_5fold/PPIGAN_5fold_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0 \
python -u ./run/train_5fold.py \
--cuda \
--interaction_data "$INTERACTION_DATA" \
--sequence_data "$SEQUENCE_DATA" \
--save_dir "$SAVE_DIR" \
--n_splits 5 \
--epoch 50 \
--batch_size 384 \
--d_steps 2 \
--g_steps 1 \
--d_lr 0.0001 \
--g_lr 0.0001 \
--beta_real_loss 1.0 \
--beta_fake_loss 0.005 \
--freq_warmup_epochs 5 \
--lambda_freq 20.0 \
--noise_scale 0.3 \
--threshold 0.5 \
--seed 42 \
--save_interval 1 \
--log_interval 20 \
> "$LOG_DIR/train.log" 2>&1 &

echo "Started 5-fold training"
echo "PID: $!"
echo "SAVE: $SAVE_DIR"
echo "LOG: $LOG_DIR/train.log"
