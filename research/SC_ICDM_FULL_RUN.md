# SC-ICDM full-scale run on sheng

## Status

Started on 2026-09-15. The resumable pipeline is running in the background on sheng.

- Remote repository: `/home/sheng/ICDM_SC_20260915`
- Pipeline log: `/mnt/d/ICDM_SC_runs/phase1_full_pipeline.log`
- PID file: `/mnt/d/ICDM_SC_runs/phase1_full_pipeline.pid`
- Dataset root: `/mnt/d/ICDM_SC_data_full`
- Download cache: `/mnt/d/ICDM_SC_downloads`
- Training output: `/mnt/d/ICDM_SC_runs/phase1_full`
- Evaluation output: `/mnt/d/ICDM_SC_runs/phase1_full_eval`

## Dataset

| Role | Train | Validation | Test |
|---|---:|---:|---:|
| Desired signal: CelebA | 162,770 | 19,867 | 19,962 |
| Interference: CIFAR-10 | 50,000 | 5,000 | 5,000 |

CelebA uses its official train/validation/test splits. The official CIFAR-10 test split is deterministically divided into 5,000 validation and 5,000 final-test images.

## Training schedule

| Component | Epochs | Batch |
|---|---:|---:|
| Interference SwinJSCC | 100 | 20 |
| Desired SwinJSCC | 40 | 20 |
| ICDM-S DiT | 50 | 20 |
| ICDM-Z DiT | 20 | 20 |

Training uses AWGN at 20 dB, channel width C=8, 128x128 desired images, the published full DiT architecture, bf16 autocast, Adam/AdamW at 1e-4, and warmed EMA with raw checkpoints retained. All stages save resumable state.

## Evaluation matrix

- 200 independent test image pairs
- Three random seeds: 0, 1, 2
- SNR: 20 dB
- SINR: -7, -4, 0, 4, 7, and 10 dB
- Interference profiles: stationary, alternating, and burst
- Modes: NO_ICDM, A0, A1, A2, and A3
- UniPC: 40 steps, order 3, noise prediction
- Evaluation batch size: 4

The batch-four path was checked end to end with the 2,000-image checkpoints. It completed without an out-of-memory error and reduced measured ICDM receiver time from about 4.4 seconds per image to about 1.1 seconds per image on the RTX 4090.

Expected wall time is roughly 40–50 hours for training plus 9–12 hours for the full evaluation. Dataset download and extraction may add several hours on the Windows-mounted D drive.
