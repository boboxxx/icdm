#!/bin/bash
#SBATCH --job-name=icdm-moments-val
#SBATCH --partition=general
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/validation-%j.out
#SBATCH --error=logs/validation-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/ICDM_SC_VTC2027
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ICDM_TEST_DEVICE=cuda:0
ICDM_PYTHON=/mnt/nfs2/engdes/wc296/venvs/hitac-rtx-cu128/bin/python
test -f assets/verified.json
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$ICDM_PYTHON" -c 'import torch; name=torch.cuda.get_device_name(); assert "6000" in name and "PRO" in name.upper(), "Requested RTX PRO 6000, got "+name'
"$ICDM_PYTHON" -m unittest discover -s tests -p test_moments.py -v
"$ICDM_PYTHON" -m scripts.run_moment_validation --manifest configs/moment_smoke.json --output results/moment_smoke_v1 --resume
"$ICDM_PYTHON" -m scripts.analyze_moment_validation --run results/moment_smoke_v1
"$ICDM_PYTHON" -m scripts.run_moment_validation --manifest configs/moment_validation.json --output results/moment_validation_v1 --resume
"$ICDM_PYTHON" -m scripts.analyze_moment_validation --run results/moment_validation_v1
