# Research decision log

| Date | Type | Decision/evidence |
|---|---|---|
| 2026-09-19 | bootstrap | User authorized rigorous ICDM modeling and experiments on artemis RTX resources for VTC2027. Preserve historical results; audit before extending. |
| 2026-09-19 | compute | Slurm uses gpu:RTX:1; shared CUDA 12.8 PyTorch 2.7.1 environment has required dependencies. No ICDM project found in the first four levels of the account home. Created a separate ICDM_SC_VTC2027 directory. |
| 2026-09-19 | authorization | User explicitly approved moving this project's trained checkpoints and required image data from sheng to their artemis account after automatic approval initially rejected transfer. |
| 2026-09-19 | protocol | Lock analytic moment tests and validation-only ablation before their execution. A5's covariance is exact for a declared Gaussian surrogate only. |
| 2026-09-19 | audit | All historical keys, image pairing, checkpoint hashes and A3/A4 initial powers agree. Image-cluster bootstrap A4−A3: +0.599 dB, CI [0.551,0.648]; severe burst −7/−4 dB differences remain negative. |
| 2026-09-19 | staging | Authorized 1,656 selected validation/test image files transferred; five checkpoint transfer still in progress. Source inventory contains 1,661 files and 5,554,371,451 bytes. No test evaluation scheduled. |
| 2026-09-19 | compute | Submitted math GPU job 11397374, CPU asset/hash gate 11397384, then GPU smoke+validation job 11397387 with afterok dependencies on both gates. At most one GPU job active, bounded 8 hours. Checks actual card name contains RTX PRO 6000 family identifiers before evaluation. |
| 2026-09-19 | implementation | Gaussian algebra, bounded optimizer and all three new sampler paths pass CPU tests; original upstream sampler remains bitwise equivalent in legacy test. Added provenance hashes and paired analysis, no retraining. |
