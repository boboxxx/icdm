# ICDM research findings

## Current understanding

Historical A4 improved mean PSNR over A3 by 0.599 dB across 200 image pairs and repeated conditions. Its power estimates often worsened and occasionally became extreme. This is evidence of useful receiver feedback in the evaluated implementation, not evidence of a calibrated posterior or of accurate physical power recovery.

The completed pairing audit found identical keys, weights, image paths and initialization. A 10,000-resample image-pair-cluster bootstrap gives +0.599 dB [0.551,0.648]. Burst −7 and −4 dB contrasts remain negative: −0.174 [−0.224,−0.126] and −0.109 [−0.197,−0.025]. These are exploratory stratum intervals, not multiplicity-controlled confirmation.

New Gaussian-surrogate calibration is implemented, with fixed-q descent and end-to-end sampler CPU tests. No new pretrained-image A5 result exists yet. Artemis jobs are submitted with integrity/math gates; the 8-hour validation job must not be mistaken for a completed experiment.

## Evidence corrections required

- Resample whole image-pair clusters, preserving seeds/SINR/profiles. The historical record-level normal intervals underestimate dependence.
- Original 200 test pairs were inspected during method development (including unit-power normalization smoke); treat them as exploratory for subsequent algorithm selection.
- Small denominators are a plausible explanation of the power outliers, not a measured causal diagnosis: historical records did not log denominators or trajectories.
- Unit-power projection fixes a scale convention but changes posterior means and uncertainty. A5 must explicitly identify its Gaussian surrogate and distinguish it from exact learned-prior posterior moments.
- Simulated `true_global_sinr` specifies nominal average amplitude power. Actual per-frame interference energy varies with block latent energy; log both in new experiments.
- Previous runtime ratio compares separate runs; benchmark methods interleaved on the same GPU before final overhead claims.

## Locked constraints

Preserve historical A0–A4 code paths. New methods receive y, SNR/noise, and block partition only. Compare A4-R, A5 diagonal-only, and A5 full cross covariance with identical damping, anchor and cap. No automatic retraining or test-guided search. Negative results remain in the report.

## Open questions

Does the A4 gain survive a better controlled validation/test protocol? Is covariance useful beyond a ridge anchor? Are amplitude changes or guidance-table changes responsible? Can a blind validation-selected bypass fix high-SINR losses?
