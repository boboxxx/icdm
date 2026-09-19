#!/bin/bash
#SBATCH --job-name=icdm-math-check
#SBATCH --partition=short
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/math-%j.out
#SBATCH --error=logs/math-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/ICDM_SC_VTC2027
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ICDM_TEST_DEVICE=cuda:0
ICDM_PYTHON=/mnt/nfs2/engdes/wc296/venvs/hitac-rtx-cu128/bin/python
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$ICDM_PYTHON" -m unittest discover -s tests -p test_moments.py -v
