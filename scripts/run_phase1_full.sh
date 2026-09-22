#!/usr/bin/env bash
set -euo pipefail

cd /home/sheng/ICDM_SC_20260915
PYTHON=/home/sheng/anaconda3/envs/cv/bin/python
DATA=/mnt/d/ICDM_SC_data_full
CACHE=/mnt/d/ICDM_SC_downloads
RUN=/mnt/d/ICDM_SC_runs/phase1_full

mkdir -p "$DATA" "$CACHE" "$RUN"

"$PYTHON" -m scripts.prepare_phase1_full \
  --root "$DATA" \
  --cache "$CACHE"

"$PYTHON" -m scripts.train_phase1_2k \
  --data-root "$DATA" \
  --output "$RUN" \
  --device cuda:0 \
  --seed 20260915 \
  --snr 20 \
  --codec-batch 20 \
  --dit-batch 20 \
  --codec-epochs 40 \
  --inf-codec-epochs 100 \
  --icdm-s-epochs 50 \
  --icdm-z-epochs 20

"$PYTHON" -m scripts.run_phase1 \
  --manifest configs/phase1_full_eval.json \
  --output /mnt/d/ICDM_SC_runs/phase1_full_eval \
  --device cuda:0
