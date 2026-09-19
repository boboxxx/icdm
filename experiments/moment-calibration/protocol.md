# Locked protocol: rigorous calibration, version 1

Date: 2026-09-19. Commit this file before running new experiments. Historical test results are exploratory. No claim that a positive result is guaranteed.

## Question and contrasts

H0: A4 historical gain survives image-pair-cluster intervals. Check all key sets, identity paths, checkpoint hashes and initial-power equality.

H1/H2: Test whether joint Gaussian surrogate moments improve over equally regularized point calibration. Main contrast A5_FULL minus A4_R. Ablations A5_DIAG minus A4_R and A5_FULL minus A5_DIAG. A3 and legacy A4 remain references; NO_ICDM anchors the weak-interference boundary. A5_DIAG uses identical conditioned means/variance but deletes the x-z covariance term; it is an ablation, not exact EM.

## Algebra and implementation gates

1. Compare scalar Gaussian conditioning to full matrix Gaussian conditioning, using float64 and complex-to-real layout tests.
2. Check nonnegative amplitude optimizer against a grid minimum of the fixed-q expected residual with anchor penalty, including nonzero cross covariance.
3. Confirm full update reduces that fixed-q quadratic under box projection and convex damping. This does not assert monotone PSNR or monotone true diffusion evidence.
4. Verify zero uncertainty reduces to stabilized point update, tail blocks, batch separation, finite outputs and trace logging.
5. Preserve original A0–A4 numerical tests. Verify RTX actual name and CUDA operation under Slurm, never on login nodes.

## Fixed implementation, no tuning in first run

AWGN only. Frozen historical encoder/decoder/priors and checkpoint SHA256. 40 UniPC steps, order 3, alpha >= 0.5 for 21 updates. A4_R/A5_DIAG/A5_FULL: amplitude cap 8, damping 0.25, anchor to A3 initial amplitude, ridge per real coordinate 0.05. All three use the same cap/anchor/damping. The cap is an a priori engineering support assumption, not inferred from a test label. Report its limitation. Guidance table is updated for all three equally. Gaussian surrogate prior variance per real coordinate is v_t=(0.5*sigma_t^2)/(0.5*alpha_t^2+sigma_t^2); same variance for x,z. Denoiser means are framewise unit-power projected to match historical A4, explicitly a modeling approximation.

## Data and phases

- Datasets: existing CelebA val and CIFAR-10 val splits for development; frozen prior weights. These val splits were also used in model training selection and are not independent confirmation.
- Smoke: first 4 val pairs, seed 700, SINR -7/0/10, profiles stationary/alternating/burst, blocks 64; all six receivers. Engineering check only, 216 records.
- Validation: first 64 val pairs, seeds 700/701, six SINRs -7/-4/0/4/7/10, three profiles, blocks 64, same six receivers (13,824 records). All configurations fixed above; no parameter sweep yet.
- Future confirmation (only after validation reviewed and config locked): 500 distinct test pairs from sorted indices 200..699 (outside old first 200), three fresh seeds 900..902. Enumerate data hashes and prove no train/val overlap first. Do not execute confirmation automatically as part of the initial script.
- Later robustness: block sizes 16/256, randomized/offset burst boundaries and duty cycles, nominal versus measured interference energy, and amplitude-only versus guidance-only controls. These need separate protocol commits before running.

## Endpoints and inference

Primary: per-image PSNR difference, average within each image over seeds and conditions; bootstrap image-pair clusters, 10,000 resamples, fixed seed 20260919. Balanced profile/SINR means. Report full CI and effect, never just significance. Secondary: matched MSE, nominal amplitude-power RMSE, actual received interference energy, per-block update denominators, cap/floor fractions, maximum power, residual energies and runtime. Keep image clusters intact for every comparison. Stratum intervals exploratory, no blanket multiplicity claim.

Claim gate: take A5 forward only if validation A5_FULL minus A4_R mean is positive with image-cluster 95% CI excluding zero, without worse low-SINR burst mean or numeric failure. Otherwise record negative result and evaluate the simpler stabilized/gated receiver. A5 is not justified merely because A4 had outliers. A gate needs validation selection against direct decoding, not true-SINR lookup at test time.

## Compute limits and artifacts

At most one Slurm RTX GPU job at once. Initial smoke+validation wall time <=8h, no retraining. Checkpoint and source hashes, resolved manifests, ordered image file hashes, actual GPU name, logs and raw records retained. Restart only if manifest/source/weights/data hashes match; do not mix incompatible partial runs. No jobs from other projects are changed.
