#!/bin/bash

set -e

INTERACTION_DATA="./data/virus-human/protein.actions.tsv"
SEQUENCE_DATA="./data/virus-human/protein.dictionary.tsv"

D_PTH="./Result/D_best_acc.pth"

mkdir -p "./Result/independent"

SAVE_DIR="./Result/independent/PPIGAN_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0 \
python -u ./run/independent.py \
--cuda \
--interaction_data "$INTERACTION_DATA" \
--sequence_data "$SEQUENCE_DATA" \
--d_pth "$D_PTH" \
--save_dir "$SAVE_DIR" \
--batch_size 64 \
--threshold 0.1 \
> "$LOG_DIR/test.log" 2>&1 &

echo "Started independent test"
echo "PID: $!"
echo "SAVE: $SAVE_DIR"
echo "LOG: $LOG_DIR/test.log"
