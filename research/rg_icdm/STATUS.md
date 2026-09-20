# RG-ICDM execution status

2026-09-20. New direction: hierarchical pooling, real receiver bypass and cross-coordinate trust feedback. The local project mirror ICDM_A4 is the active implementation; Desktop/ICDM contains the older A3 version.

Protocol acaa5d1 was committed before execution. Implementation 95cfc7f adds the receiver, runner, analysis and tests. The small integration change to Diffusion/uni_pc.py sits on pre-existing user edits and is captured exactly in each run source archive; it was not committed together with unrelated changes. All old A0–A5 branches remain available. CPU checks: 3 new receiver tests, 5 old moment regression tests and 13 historical A0–A4 regression tests passed (21 total).

Artemis job uses an isolated ICDM_RG_20260920 directory with the original verified assets from ICDM_SC_VTC2027. Slurm receipt: **11399316**, pending resources at first inspection. It requests one RTX GPU, 4 CPUs, 32 GB RAM, maximum 8 hours. Model-name assertion requires PRO and 6000. Pipeline: GPU tests → 96-record engineering smoke → strict analysis → 18,432-record validation → strict analysis. No retraining.

Historical phase-0 oracle routing headroom is 0.861065 dB over A3 on 10,800 paired trials. It is an analysis upper reference, not achieved blind routing performance. Prior A5_FULL validation was complete, and its paired mean against A4_R was −0.600463 dB.

No new RG performance claim yet. Metrics in the first run are PSNR, MSE, harm fraction, lower tail, NFE and elapsed time. Artemis currently has no lpips, transformers or open_clip in this runtime and no verified cached perceptual weights. Perceptual and semantic metrics remain explicit follow-up work, not silently replaced by pixel metrics.

## Completed RG validation

Slurm job **11399316** completed successfully on an NVIDIA RTX PRO 6000 Blackwell Server Edition in 2:57:36. The run produced all 18,432 prospectively specified validation records. GPU tests and provenance checks passed.

The strongest method was B_COUPLED at 20.5773 dB. Relative to A3 it gained 0.5600 dB on average, won 76.65% of paired conditions, reduced the rate of losses larger than 0.5 dB to 6.29%, and had a fifth-percentile paired change of -0.6617 dB. B3, which adapts amplitude while freezing initial guidance, gained 0.4414 dB over A3. B_GUIDE gained 0.3381 dB over A3. This supports both amplitude feedback and coupled control; guidance-only adaptation was weaker.

B2 bypassed 17.19% of frames overall, including all 10 dB frames and 3.125% of 7 dB frames. It reduced mean prior NFE from 80 to 66.25. At 10 dB, B2/B3/B_GUIDE/B_COUPLED exactly matched direct decoding at 27.3225 dB, while A3 reached 24.9739 dB. The bypass did not activate from -7 through 4 dB.

The result is developmental validation on a previously inspected split. It supports continuing the RG-ICDM direction but is not independent confirmation. Archived summaries are `rg_validation_analysis.json` and `rg_validation_metadata.json`. The remaining scientific work is frozen configuration selection, robustness to unaligned/random and same-distribution interference, verified perceptual/semantic metrics, and evaluation on an untouched confirmation split.

The icdm-artemis heartbeat checks follow-up progress every 20 minutes and stays quiet when unchanged. Consult STATE.json for the latest stage.
