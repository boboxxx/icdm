# SC-ICDM 2,000-image feasibility result

## Outcome

The complete training and inference chain runs on sheng. With final raw diffusion weights, every ICDM variant improves reconstruction over decoding the interfered signal without ICDM. The blind blockwise mode (A3) gives the largest gain under alternating and burst interference in this small evaluation.

These results establish engineering feasibility. They are based on four validation image pairs, one seed, and one SNR/SINR point, so they do not establish statistical superiority.

## Setup

- Host: sheng, NVIDIA GeForce RTX 4090
- Desired data: 2,000 CelebA training images and 200 independent validation images
- Interference data: 2,000 CIFAR-10 training images and 200 independent test images
- Channel: AWGN
- Evaluation: SNR 20 dB, global SINR 0 dB, block size 64
- Profiles: stationary, alternating, and burst interference power
- Sampling: 40 UniPC steps, order 3, noise prediction
- Evaluation set used below: four image pairs, seed 0
- Total training wall time: 1.004 hours

## Reconstruction results

Mean PSNR in dB. Parentheses show the gain over `NO_ICDM` for the same profile.

| Mode | Stationary | Alternating | Burst |
|---|---:|---:|---:|
| NO_ICDM | 15.053 | 15.640 | 16.303 |
| A0 original oracle | 19.370 (+4.317) | 18.977 (+3.337) | 18.514 (+2.211) |
| A1 power-consistent oracle | 19.372 (+4.319) | 18.975 (+3.336) | 18.516 (+2.213) |
| A2 blind global estimate | **19.411 (+4.358)** | 19.039 (+3.399) | 18.608 (+2.305) |
| A3 blind blockwise estimate | 19.072 (+4.019) | **19.146 (+3.507)** | **19.264 (+2.962)** |

Relative to A0, A3 changes PSNR by -0.298 dB for stationary interference, +0.170 dB for alternating interference, and +0.750 dB for burst interference. This is the expected qualitative pattern for a method designed for nonstationary interference, but it must be confirmed on a larger evaluation.

A0 and A1 are effectively identical at SNR 20 dB and SINR 0 dB. At this operating point the two amplitude formulas are numerically close, so this run cannot resolve the power-consistency question.

## Runtime

Mean receiver time per image on the RTX 4090:

| Mode | Time (s) |
|---|---:|
| NO_ICDM | 0.245 |
| A0 | 4.436 |
| A1 | 4.406 |
| A2 | 4.391 |
| A3 | 4.424 |

The ICDM time includes calibration, 40-step sampling, normalization, and decoding. It excludes source encoding, channel generation, and data loading. Calibration adds no material runtime relative to oracle ICDM.

## Training diagnostics

| Component | Epochs | Final validation metric |
|---|---:|---:|
| Interference codec | 40 | MSE 0.000252; PSNR 35.986 dB |
| Desired codec | 40 | MSE 0.002052; PSNR 26.879 dB |
| ICDM-S | 50 | EMA validation noise loss 0.567972 |
| ICDM-Z | 20 | EMA validation noise loss 0.877928 |

The best desired-codec validation MSE was 0.001924 at epoch 38.

## EMA finding

The upstream fixed EMA decay of 0.9999 is unsuitable for this short run. After roughly 5,000 ICDM-S updates it retains about 60.6% of the initial random model; after roughly 2,000 ICDM-Z updates it retains about 81.9%. Evaluation with those cold EMA weights produced only about 6.8–8.2 dB, below `NO_ICDM`.

The reported reconstruction table therefore uses the final raw ICDM-S and ICDM-Z weights. The training script now warms up the EMA decay and automatically saves both EMA and raw weights for future runs.

## Calibration diagnostics

True global interference power is 0.99. A2 estimates mean powers of 0.890, 0.905, and 0.939 for stationary, alternating, and burst profiles. A3 block estimates are noisy: mean absolute block-power error is 0.630, 0.648, and 0.662 respectively. Despite this, A3 improves reconstruction under both nonstationary profiles in the four-image test.

This suggests the next priority is to evaluate more images and seeds, then improve the block estimator or proceed to the alternating update in A4. A3's current block estimates are not accurate enough to treat its apparent advantage as final evidence.

## Reproducibility artifacts

- Training log: `research/sheng_phase1_2k/phase1_2k/training.jsonl`
- Complete raw-weight evaluation: `research/sheng_phase1_2k/phase1_2k_raw_eval_4/records.jsonl`
- Evaluation metadata and source hashes: `research/sheng_phase1_2k/phase1_2k_raw_eval_4/metadata.json`
- Cold-EMA diagnostic records: `research/sheng_phase1_2k/phase1_2k_eval_10/records.jsonl`
- Remote weights: `/mnt/d/ICDM_SC_runs/phase1_2k/weights`

