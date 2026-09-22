# Prospective VTC 2027 receiver study on sheng

Revised protocol frozen 2026-09-21 before formal quality analysis and independent confirmation. The initial partial development run is excluded; see SELECTOR_AMENDMENT.md.

Question: does choosing a Gaussian disturbance model or an explicit interference
prior improve frozen-codec image reception under unknown interference power?
The initial study is not a calibrated-risk or semantic-task guarantee.

Assets: original five phase1_full checkpoints, checked against the prior RG
SHA256 manifest. Same 128x128 CelebA target images and CIFAR-10 interference
encoder (32x32 crop then nearest upsample), 8-channel latent, AWGN SNR 20 dB.
Receiver knows N0 and the codec normalization, not true SINR/profile/power.
All methods receive exactly the same y. All stochastic samplers reset both CPU
and GPU random states to the same sample-specific sampler seed. Channel and
sampler random streams are separated. Float32, TF32 off, batch one, 4 CPU threads.

Development: the previously inspected first 64 validation pairs, one NEW seed
1700, SINR {-7,-4,0,4,7,10}, profiles stationary/alternating/burst, blocks 64.
Modes: DIRECT, CDDM_N0, CDDM_BLIND, A3, A4, B_COUPLED, CDDM_ORACLE.
Oracle is a diagnostic with privileged nominal total disturbance power.
CDDM remains a shared-DiT receiver port (40 evaluations), not its full retrained
published system. A4_GATE is a deterministic composition of the recorded
DIRECT/A4 outputs with exactly the previous 0.15 estimated interference-power
threshold. Its quality can be replayed exactly; timing is a component estimate
until direct online verification. No extra denoising is required to select it.

Prospective selection family: CDDM_BLIND versus A4; total estimated interference
power thresholds {0,.1,.2,.35,.5,.75,1,1.5,2,3,4,8,1e6}. A two-feature extension
uses squared coefficient of variation of raw block energy (variance divided by
squared frame energy), heterogeneity thresholds {.05,.1,.2,.4,.8,1.6}, and a
separate power threshold on each side. Both Gaussian-below and Gaussian-above threshold directions are candidates (26 energy-only and 2,028 two-feature rules). Direct decoding stays an explicit
baseline; do not add extra actions after viewing confirmation data.
First compare simple one-feature and two-feature selectors using 8 image-group
folds (index mod 8), fit on seven folds, predict only the omitted fold. Objective
is mean per-image PSNR; ties prefer lower estimated inference cost and then
one-feature complexity. Report LPIPS, mean MSE and harm without optimizing them
on confirmation. Final selector selection and fitted rule must be frozen in a
separate policy.json before confirmation is opened. No claim of a new
statistical risk-control theorem is made.

Confirmation reservation: 256 pairs from official test folders, sorted indices
[512,768), two NEW seeds 2700/2701, same six SINRs and three profiles. Original
test evaluations used the first 200 images; do not reuse these as confirmation.
Use image-hash checks across development/confirmation and original recorded
evaluation manifests. Official train/val/test source splits remain separate;
without a separate identity audit, claim image-level, not identity-level or
population-wide independence. Confirmatory modes are DIRECT, CDDM_N0,
CDDM_BLIND, A3, A4, plus the frozen online selector. B_COUPLED is only included
if development warrants its extra computation; it is not the default main
method. This smaller study than the earlier advisory plan avoids an unsupported
risk certificate and fits one available GPU. No certified-risk claim is planned.

Pressure tests after rule freeze: 64 additional test pairs [1024,1088), seed
3700, SINR {-4,4,10}, shifted block boundaries and random-length block patterns;
also no-interference SNR20. Keep each pressure family separate from the matched
confirmation population. Include optional same-source interference only with
an explicit source-prior mismatch/identifiability caveat.

Metrics: clipped RGB PSNR/MSE, LPIPS-VGG v0.1, audited 4-scale MS-SSIM, individual
signal/interference NFE, receiver seconds (synchronized, warmed, batch one,
excluding metrics/encoding), losses exceeding 0.5 dB and fifth-percentile
paired change. Image-cluster paired bootstrap intervals; seeds/conditions from
one image are not independent replicates. Qualitative indices are fixed before
inspection. No cherry-picking.

Engineering smoke is excluded from scientific results. A warm-up repeat must
match with reset seeds. Source/weights/data/metric hashes, environment, GPU
occupation, status and resumable unique record keys are saved. No other user's
process is terminated. Failure and negative results change the manuscript
claims, not the recorded data. Publishing/submitting is outside this execution.
