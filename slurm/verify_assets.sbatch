#!/bin/bash
#SBATCH --job-name=icdm-assets
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/assets-%j.out
#SBATCH --error=logs/assets-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/ICDM_SC_VTC2027
export PYTHONDONTWRITEBYTECODE=1
/mnt/nfs2/engdes/wc296/venvs/hitac-rtx-cu128/bin/python -m scripts.verify_artemis_assets --root assets --inventory configs/artemis_asset_inventory.json
