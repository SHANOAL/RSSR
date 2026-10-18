#!/usr/bin/env bash
# Train CrossVAR with the recipe of the paper: one image per GPU, three GPUs.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train.sh INIT_CKPT [extra train.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
INIT=${1:?initialization checkpoint}; shift
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
torchrun --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-29700}" train.py \
  --exp_name crossvar --train_data_path datasets/train --valid_data_path datasets/val \
  --batch_size 1 --workers 4 --ep 6 --epoch_snapshot_interval 1 --save_ce_best False \
  --head_lr_scale 5 --condition_lr_scale 5 \
  --stage_loss_power 0.5 --img_loss_w 1 --l1_w 1 --ssim_w 0.05 --diff_loss_w 2 --ste_tau 1 --ste_topk 16 \
  --pn 1_2_3_4_6_9_13_18_25 --depth 24 --fp16 2 --tblr 2.5e-7 --ac 1 --wp 0.3 --wpe 0.1 \
  --sche cos --twd 0.01 --twde 0.01 --tclip 1 --seed 3407 \
  --vae_model_path "$INIT" --var_pretrain_path "$INIT" "$@"
