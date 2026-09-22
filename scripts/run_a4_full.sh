#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/ICDM_SC_20260915
python=/home/sheng/anaconda3/envs/cv/bin/python
output=/mnt/d/ICDM_SC_runs/a4_full_eval
resume=()
if [[ -d "$output" ]]; then
  resume=(--resume)
fi
PYTHONDONTWRITEBYTECODE=1 "$python" -m scripts.run_phase1 \
  --manifest configs/a4_full_eval.json \
  --output "$output" \
  "${resume[@]}"
