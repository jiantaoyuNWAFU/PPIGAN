#!/bin/bash

set -e

# ===== 数据路径 =====
INTERACTION_DATA="./data/Negatome2/protein.actions.tsv"
SEQUENCE_DATA="./data/Negatome2/protein.dictionary.tsv"

mkdir -p "./fuse_Result/Negatome2"

# ===== 输出路径 =====
SAVE_DIR="./fuse_Result/Negatome2/PPIGAN_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# ===== 训练 =====
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u ./run/train_fuse.py \
--cuda \
--interaction_data "$INTERACTION_DATA" \
--sequence_data "$SEQUENCE_DATA" \
--save_dir "$SAVE_DIR" \
--batch_size 128 \
--g_steps 2 \
--beta_fake_loss 0.05 \
--lambda_freq 5.0 \
--freq_warmup_epochs 5 \
--noise_scale 1.0 \
--seed 42 \
> "$LOG_DIR/train.log" 2>&1 &

echo "Started training"
echo "PID: $!"
echo "SAVE: $SAVE_DIR"
echo "LOG: $LOG_DIR/train.log"