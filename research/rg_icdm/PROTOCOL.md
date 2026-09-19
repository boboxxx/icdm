# RG-ICDM prospective validation protocol — 2026-09-20

Research question: can receiver-only evidence avoid harmful generative intervention while retaining gains under nonstationary interference? A5 is retained as a negative diagnostic, not the main hypothesis. Reuse all five verified checkpoints without retraining.

## Scientific model and limits

AWGN complex channel y=x+a*z+w; unit mean complex source/interferer power and known N0=10^(-SNR/10). The receiver gets y, h=1, N0 and its own fixed block size, never simulator profile, true SINR, true power, source image or clean latent. A fixed receiver partition is not knowledge of random change points.

For each block use untruncated energy excess q_b and sample-variance-of-mean s_b². Estimate global q_g using block lengths. With weights u_b=L_b/N, use tau²=max(0, [sum u_b(q_b-q_g)²-sum u_b(1-u_b)s_b²]/[1-sum u_b²]). The B1 estimate is max(0,q_g)+w_b(max(0,q_b)-max(0,q_g)), w_b=tau²/(tau²+s_b²). This is an empirical random-effects heuristic: symbol correlation and the neural latent distribution invalidate any claim of an exact posterior variance.

B2 bypasses when estimated global interference power <=0.15 (initial engineering threshold, not a validated optimum). B3 adds late (alpha>=0.8) ridge-anchored amplitude proposals, damping 0.25 and absolute step limit 0.25. Fit on even complex symbols and assess a clipped proposal's residual improvement on odd symbols, scaled by N0+sigma²/alpha². Require a positive margin 0.05 and adequate interference energy in both subsets. Because denoisers use all y, this is a cross-coordinate consistency diagnostic, NOT independent held-out evidence, a hypothesis test, or a semantic guarantee. Maximum amplitude 8.

## Causal controls

NO_ICDM; A2 (global); A3/B0 (local); B1 (hierarchical, no bypass); B2 (B1+bypass); B3 (B2+amplitude feedback, guidance fixed at initialization); B_GUIDE (B2+guidance feedback, amplitude fixed); B_COUPLED (both adapt). No change to historical A0–A5 paths.

Use batch size one, common per-image/channel/sampler seeds and deterministic kernels so conditional bypass never shifts random initialization of nonbypassed frames. Count both prior forward calls with hooks; bypass NFE=0. Time actual receivers with GPU synchronization, excluding perceptual scoring. Include warmup.

## Stages and locked endpoints

1. Phase 0: strictly pair historical NO_ICDM/A2/A3 records, compute max-per-trial oracle routing headroom relative to best fixed method. Oracle is analysis only; historical test results remain exploratory.
2. CPU/GPU tests: tau=0 pooling, nonuniform/short blocks, no target leakage interface, zero denominator, rejected harmful update, bounded accepted update, fixed guidance, genuine NFE-zero bypass, batch-one paired equivalence.
3. Smoke: 2 validation image pairs, seed 700, SINR -7/0/10, stationary/burst. Engineering verification only.
4. Validation: 64 existing validation pairs, seeds 700/701, SINR -7/-4/0/4/7/10, stationary/alternating/burst; all eight modes. This is developmental validation already used in prior work, not fresh confirmation. No large seed sweep.
5. Report mean PSNR, paired delta, harm fraction delta<-0.5 dB, fifth-percentile delta, per-condition results, NFE and runtime. LPIPS only when pretrained metric weights are available and hashed; MS-SSIM is a structural metric, not semantic consistency. Semantic feature metric remains pending until its pretrained model is verified.
6. Before subsequent evaluation freeze a validation-selected gate/config with source and data hashes. Random piecewise unaligned segments and same-source-distribution interference are subsequent robustness stages. Never present already inspected 200 historical test pairs as independent confirmation. A new untouched split is required for confirmatory claims.

## Decisions

B1 vs A3 tests shrinkage. B2 vs B1 tests bypass. B3 vs B2 tests trusted amplitude feedback. B_GUIDE and B_COUPLED isolate guidance from physical amplitude. If amplitude-only fails, retain negative result and reconsider guidance control rather than claiming calibration improvement. Stop A5 as mainline. Do not claim a successful paper before results. One RTX PRO 6000 allocation maximum; no unrelated cancellations.
