# ICDM-Interference-Cancellation-Diffusion-Models-for-Wireless-Semantic-Communications
This is the official Pytorch implement of the paper "ICDM：Interference Cancellation Diffusion Models for Wireless Semantic Communications"

## Code release

This repository contains the model, diffusion sampler, receiver-side calibration code, evaluation scripts, tests, and the VTC 2027 system-model notes. Datasets, trained weights, raw experiment records, and machine-specific execution artifacts are intentionally excluded. Configure local dataset and checkpoint paths before running an evaluation.

The controlled moment-calibration extension is documented in `research/SYSTEM_MODEL_VTC2027.md` and `research/VTC2027_EXECUTION_PLAN.md`. The receiver-side update uses only the received latent, noise level, and known block partition; the simulated SINR and clean latents are not receiver inputs.

## Completed VTC 2027 Spring study (2026-09-23)

The reviewed five-page paper is [VTC2027_Sheng_Manuscript.pdf](output/vtc2027_sheng/VTC2027_Sheng_Manuscript.pdf), with [LaTeX source](output/vtc2027_sheng/VTC2027_Sheng_LaTeX.zip) and [Chinese results and limitations](output/vtc2027_sheng/RESULTS_AND_REVIEW_ZH.md).

Table I/Figure 1 include matched-retrained DeepJSCC and official MambaJSCC direct-codec anchors (18,432 records). Section IV-B additionally reports three public-checkpoint diagnostics without fine-tuning (27,648 records). The core claim remains the frozen-SwinJSCC selector comparison against blind Gaussian, calibrated joint, and same-gate controls. The external anchors do not reproduce complete published CDDM/ICDM systems or establish an architecture-wide SOTA ranking. See [reproduction instructions](paper/vtc2027_selective_sheng/README.md). Author information remains explicitly pending; no submission has been made.
