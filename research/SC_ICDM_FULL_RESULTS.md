# SC-ICDM full-scale evaluation result (A0–A4)

Audit correction (2026-09-19): historical data were inspected during development and are exploratory. A4 confidence intervals below now resample image-pair clusters; previous record-level normal intervals are withdrawn. New A5 experiments are pending, not included here.

## Completed evaluation

The combined experiment completed on `sheng` with 64,800 reconstruction records: 54,000 records for `NO_ICDM` and A0–A3, plus 10,800 strictly matched A4 records.

- 200 CelebA test-split images with CIFAR-10 interference images, subsequently used for development
- three channel seeds per image
- SNR = 20 dB and SINR = -7, -4, 0, 4, 7, 10 dB
- stationary, alternating, and burst interference profiles
- `NO_ICDM`, A0 (original global-parameter oracle), A1 (power-consistent global oracle), A2 (blind global estimate), A3 (blind blockwise initialization), and A4 (alternating point-estimate block calibration)
- 40-step, order-3 UniPC sampling with an evaluation batch size of four

Every method has 10,800 results, and every profile-level cell below averages 3,600 paired reconstructions. A4 uses the same image, interference-image, channel-noise, and diffusion-noise keys as A0–A3.

## Mean reconstruction PSNR

| Interference profile | NO_ICDM | A0 oracle | A1 oracle | A2 blind global | A3 blind blockwise | A4 alternating |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 18.616 | 19.058 | 19.066 | 19.053 | 18.766 | **19.446** |
| Alternating | 18.811 | 18.939 | 18.948 | 18.931 | 19.758 | **20.352** |
| Burst | 19.046 | 18.904 | 18.913 | 18.899 | 21.012 | **21.535** |
| All profiles | 18.825 | 18.967 | 18.976 | 18.961 | 19.845 | **20.444** |

## Paired comparisons

The A4 intervals use 10,000 percentile-bootstrap resamples of 200 image-pair clusters (seed 20260919), preserving all repeated seeds/SINRs/profiles within each image pair. Stratum intervals are exploratory, without multiplicity adjustment. A2−A0 and A3−A0 below are descriptive means; their earlier normal intervals are withdrawn pending cluster analysis. These are not independent confirmation results.

| Profile | A2 − A0 | A3 − A0 | A4 − A0 | A4 − A3 |
|---|---:|---:|---:|---:|
| Stationary | -0.005 | -0.292 | **+0.388 [+0.308, +0.468]** | **+0.680 [+0.629, +0.729]** |
| Alternating | -0.008 | **+0.819** | **+1.413 [+1.318, +1.507]** | **+0.594 [+0.540, +0.649]** |
| Burst | -0.005 | **+2.108** | **+2.632 [+2.502, +2.761]** | **+0.523 [+0.469, +0.578]** |
| All profiles | -0.006 | +0.878 | **+1.478 [+1.392, +1.563]** | **+0.599 [+0.551, +0.648]** |

A4 outperforms A3 in 79.1% of all paired trials. The median paired gain is +0.597 dB.

## A4 gain relative to A3 by SINR

| SINR (dB) | Stationary | Alternating | Burst |
|---:|---:|---:|---:|
| -7 | +0.323 | +0.206 | **-0.174** |
| -4 | +0.452 | +0.001 | **-0.109** |
| 0 | +0.315 | +0.454 | +0.446 |
| 4 | +1.084 | +0.925 | +0.846 |
| 7 | +1.050 | +1.109 | +1.208 |
| 10 | +0.854 | +0.870 | +0.923 |

The alternating -4 dB interval crosses zero. The exploratory image-cluster intervals for burst differences exclude zero: -0.174 dB [-0.224, -0.126] at -7 dB and -0.109 dB [-0.197, -0.025] at -4 dB.

Across profiles, A4 exceeds A3 by +0.118, +0.115, +0.405, +0.952, +1.122, and +0.882 dB at SINR -7, -4, 0, 4, 7, and 10 dB, respectively.

## A4 point-calibration diagnostic

A4 initializes every block with the A3 energy estimate and applies 21 nonnegative least-squares point updates during the latter part of the 40-step reverse process. The latent signal and interference point estimates are constrained to unit complex power before each update.

The reconstruction gain does not imply that the final point powers are physically more accurate:

- the A3 initialization has an aggregate block-power RMSE of 0.953;
- only 28.3% of A4 records finish with a lower per-record block-power RMSE;
- A4 systematically shrinks the median final power as SINR increases: 2.761, 1.164, 0.234, 0.024, 0.009, and 0.004 at -7, -4, 0, 4, 7, and 10 dB;
- 14 records contain a final block-power estimate above 100, and the maximum is 93,186.656. Historical logs do not contain denominators, so a small-denominator cause is a hypothesis, not an observed diagnosis.

Thus, A4 supports a reconstruction gain in this sample, not consistent recovery of physical interference power. The instability motivates equally regularized point/moment controls; it does not establish that posterior-variance regularization works. The planned A5 uses Gaussian-surrogate, not exact neural-posterior, moments.

## High-SINR boundary and receiver cost

A4 improves A3 at 10 dB by +0.882 dB [+0.817, +0.947] when profiles are pooled, but A4−direct is -1.070 dB [-1.270, -0.867]. A blind validation-selected bypass remains to be tested.

| Receiver | Mean receiver time per image | Relative cost |
|---|---:|---:|
| A3 | 0.323 s | 1.000× |
| A4 | 0.347 s | 1.072× |

The separate batch-four RTX 4090 runs differ by 0.023 s per image, or 7.2%. This is descriptive, not a controlled overhead estimate; new timings interleave methods on the same GPU.

## Interpretation

1. A2 avoids true global SINR and has similar mean PSNR to A0; similarity is not formal statistical equivalence.
2. A3 establishes that local interference structure matters: a blind blockwise receiver substantially exceeds the true-global-SINR reference under alternating and burst interference.
3. A4 improves A3 by 0.599 dB overall in the exploratory sample, with image-cluster interval [0.551,0.648]; independent confirmation remains outstanding.
4. The gain is conditional. A4 loses to A3 under the two strongest burst conditions, and every ICDM variant remains inferior to direct decoding at 10 dB.
5. A4's final amplitudes are not reliable physical estimates. A5 should test whether posterior moments stabilize the update; A6 should gate local calibration and bypass unnecessary cancellation.

## Training diagnostics

| Component | Completed schedule | Best validation result |
|---|---:|---:|
| Interference SwinJSCC | 100 epochs | MSE 5.196e-05; PSNR 42.843 dB |
| Desired SwinJSCC | 40 epochs | MSE 1.039e-03; PSNR 29.832 dB |
| ICDM-S | 50 epochs | validation noise loss 0.148180 |
| ICDM-Z | 20 epochs | validation noise loss 0.146287 |

## Evidence locations

- A0–A3 metadata and 54,000 records: `research/sheng_phase1_full/`
- A4 metadata and 10,800 records: `research/sheng_a4_full/`
- Remote A0–A3 output: `/mnt/d/ICDM_SC_runs/phase1_full_eval/`
- Remote A4 output: `/mnt/d/ICDM_SC_runs/a4_full_eval/`
- A4 pipeline log: `/mnt/d/ICDM_SC_runs/a4_full_pipeline.log`
