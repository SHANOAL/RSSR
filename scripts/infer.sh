#!/usr/bin/env bash
# Complete-scene inference with aligned window decoding, then PSNR, SSIM, LPIPS, and FID.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh CKPT VQVAE_CKPT [TEST_ROOT]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?CrossVAR weights}; VAE=${2:?VQ autoencoder weights}; ROOT=${3:-datasets/test}
python infer.py --test_root "$ROOT" --var_test_path "$CKPT" --vae_model_path "$VAE" --fp16 2 --seed 3407 \
  --refine_temperature 0 --fusion rgb --overlap 5 --report_json "$ROOT/inference_speed.json"
python scripts/evaluate.py --hr-dir "$ROOT/HR" --pred-dir "$ROOT/VARPrediction" --output-json "$ROOT/metrics.json" \
  --expected-count "$(ls "$ROOT/LR" | wc -l)" --expected-size 1600 --method CrossVAR
