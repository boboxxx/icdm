# ICDM-Interference-Cancellation-Diffusion-Models-for-Wireless-Semantic-Communications
This is the official Pytorch implement of the paper "ICDM：Interference Cancellation Diffusion Models for Wireless Semantic Communications"

## Code release

This repository contains the model, diffusion sampler, receiver-side calibration code, evaluation scripts, tests, and the VTC 2027 system-model notes. Datasets, trained weights, raw experiment records, and machine-specific execution artifacts are intentionally excluded. Configure local dataset and checkpoint paths before running an evaluation.

The controlled moment-calibration extension is documented in `research/SYSTEM_MODEL_VTC2027.md` and `research/VTC2027_EXECUTION_PLAN.md`. The receiver-side update uses only the received latent, noise level, and known block partition; the simulated SINR and clean latents are not receiver inputs.
