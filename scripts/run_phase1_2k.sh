#!/usr/bin/env bash
set -euo pipefail

repo=/home/sheng/ICDM_SC_20260915
python=/home/sheng/anaconda3/envs/cv/bin/python
data=/mnt/d/ICDM_SC_data_2000
output=/mnt/d/ICDM_SC_runs/phase1_2k

cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

"$python" -m scripts.prepare_phase1_2k \
  --root "$data" \
  --celeba-parquet "$data/raw/img_align+identity+attr/train-00000-of-00019.parquet" \
  --cifar-train-parquet "$data/raw/plain_text/train-00000-of-00001.parquet" \
  --cifar-val-parquet "$data/raw/plain_text/test-00000-of-00001.parquet" \
  --train-size 2000 --val-size 200

"$python" -m scripts.train_phase1_2k \
  --data-root "$data" --output "$output" \
  --codec-epochs 40 --inf-codec-epochs 40 \
  --icdm-s-epochs 50 --icdm-z-epochs 20
