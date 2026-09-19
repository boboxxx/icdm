# Research decision log

| Date | Type | Decision/evidence |
|---|---|---|
| 2026-09-19 | bootstrap | User authorized rigorous ICDM modeling and experiments on artemis RTX resources for VTC2027. Preserve historical results; audit before extending. |
| 2026-09-19 | compute | Slurm uses gpu:RTX:1; shared CUDA 12.8 PyTorch 2.7.1 environment has required dependencies. No ICDM project found in the first four levels of the account home. Created a separate ICDM_SC_VTC2027 directory. |
| 2026-09-19 | authorization | User explicitly approved moving this project's trained checkpoints and required image data from sheng to their artemis account after automatic approval initially rejected transfer. |
| 2026-09-19 | protocol | Lock analytic moment tests and validation-only ablation before their execution. A5's covariance is exact for a declared Gaussian surrogate only. |
