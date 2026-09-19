# RG-ICDM execution status

2026-09-20. New direction: hierarchical pooling, real receiver bypass and cross-coordinate trust feedback. The local project mirror ICDM_A4 is the active implementation; Desktop/ICDM contains the older A3 version.

Protocol acaa5d1 was committed before execution. Implementation 95cfc7f adds the receiver, runner, analysis and tests. The small integration change to Diffusion/uni_pc.py sits on pre-existing user edits and is captured exactly in each run source archive; it was not committed together with unrelated changes. All old A0–A5 branches remain available. CPU checks: 3 new receiver tests, 5 old moment regression tests and 13 historical A0–A4 regression tests passed (21 total).

Artemis job uses an isolated ICDM_RG_20260920 directory with the original verified assets from ICDM_SC_VTC2027. Slurm receipt: **11399316**, pending resources at first inspection. It requests one RTX GPU, 4 CPUs, 32 GB RAM, maximum 8 hours. Model-name assertion requires PRO and 6000. Pipeline: GPU tests → 96-record engineering smoke → strict analysis → 18,432-record validation → strict analysis. No retraining.

Historical phase-0 oracle routing headroom is 0.861065 dB over A3 on 10,800 paired trials. It is an analysis upper reference, not achieved blind routing performance. Prior A5_FULL validation was complete, and its paired mean against A4_R was −0.600463 dB.

No new RG performance claim yet. Metrics in the first run are PSNR, MSE, harm fraction, lower tail, NFE and elapsed time. Artemis currently has no lpips, transformers or open_clip in this runtime and no verified cached perceptual weights. Perceptual and semantic metrics remain explicit follow-up work, not silently replaced by pixel metrics.

The icdm-artemis heartbeat checks progress every 20 minutes and stays quiet when unchanged. It must retrieve completed records, audit pairing, report negative results and continue the remaining staged work. Consult STATE.json for the receipt and latest stage.
