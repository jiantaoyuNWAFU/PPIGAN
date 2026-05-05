#!/bin/bash

set -e

# ===== 数据路径 =====
INTERACTION_DATA="./data/yeast/protein.actions.tsv"
SEQUENCE_DATA="./data/yeast/protein.dictionary.tsv"

mkdir -p "./Result/yeast"

# ===== 输出路径 =====
SAVE_DIR="./Result/yeast/PPIGAN_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ===== 训练 =====
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u ./run/train.py \
--cuda \
--interaction_data "$INTERACTION_DATA" \
--sequence_data "$SEQUENCE_DATA" \
--save_dir "$SAVE_DIR" \
--batch_size 128 \
--g_steps 1 \
--beta_fake_loss 0.005 \
--lambda_align 0.1 \
--lambda_freq 10.0 \
--noise_scale 0.1 \
--seed 42 \
> "$LOG_DIR/train.log" 2>&1 &

echo "Started training"
echo "PID: $!"
echo "SAVE: $SAVE_DIR"
echo "LOG: $LOG_DIR/train.log"