#!/bin/bash
#SBATCH --job-name=icdm-rg-validation
#SBATCH --partition=general
#SBATCH --gres=gpu:RTX:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rg-%j.out
#SBATCH --error=logs/rg-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/ICDM_RG_20260920
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ICDM_TEST_DEVICE=cuda:0
export PYTHONPATH=/mnt/nfs2/engdes/wc296/ICDM_SC_VTC2027/vendor/python${PYTHONPATH:+:$PYTHONPATH}
ICDM_PYTHON=/mnt/nfs2/engdes/wc296/venvs/hitac-rtx-cu128/bin/python
test -f /mnt/nfs2/engdes/wc296/ICDM_SC_VTC2027/assets/verified.json
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$ICDM_PYTHON" -c 'import torch; name=torch.cuda.get_device_name(); assert "6000" in name and "PRO" in name.upper(), "Requested RTX PRO 6000, got "+name; print(torch.__version__,name)'
"$ICDM_PYTHON" -m unittest discover -s tests -p test_selective.py -v
"$ICDM_PYTHON" -m unittest discover -s tests -p test_moments.py -v
"$ICDM_PYTHON" -m scripts.run_rg_validation --manifest configs/rg_smoke.json --output results/rg_smoke_v1 --resume
"$ICDM_PYTHON" -m scripts.analyze_rg --run results/rg_smoke_v1
"$ICDM_PYTHON" -m scripts.run_rg_validation --manifest configs/rg_validation.json --output results/rg_validation_v1 --resume
"$ICDM_PYTHON" -m scripts.analyze_rg --run results/rg_validation_v1
