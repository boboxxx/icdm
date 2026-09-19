# Published baseline and metric audit — 2026-09-19

## Scope and claim boundary

This is a receiver-comparison validation study, not a reproduction of published
leaderboard numbers. Existing A1–A4 are internal variants, not four external
baselines. No new held-out test is opened. The existing validation images were
used for model selection previously; results are exploratory.

The first run is fixed before execution: 64 sorted CelebA/CIFAR-10 validation
pairs, seeds 700/701, SNR 20 dB, SINR -7/-4/0/4/7/10 dB, three historical power
profiles and 64-complex-symbol blocks. All five receivers get the same channel
realization. Batch size 4; original frozen weights; no hyperparameter selection
using these outcomes. Smoke: four pairs, seed 700, SINR -7/0/10. No old result is
overwritten. Receiver time excludes quality metrics and encoding, and includes
normalization and decoding. No other server job is cancelled.

## Source-grounded choices

| Method and publication | Role in this run | Information and implementation limits |
|---|---|---|
| SwinJSCC, IEEE TCCN 2025, DOI 10.1109/TCCN.2024.3424842 | Direct frozen-codec reference (`SWINJSCC`) | Existing ICDM-project Swin codec/checkpoints, not the original paper's adaptive-rate system or reported scores. |
| CDDM, IEEE TWC 2024, DOI 10.1109/TWC.2024.3379244 | New receiver-only algorithm port (`CDDM_RX_N0`) | Algorithm 2 deterministic denoising; shared DiT prior, exact prior training schedule, frozen decoder. Thermal-noise power only. Does not model semantic interference. |
| CDDM receiver port + global oracle disturbance power (`CDDM_RX_ORACLE`) | Information-advantaged Gaussian approximation control | Same implementation, but uses total nominal noise-plus-interference power. Not another published method; no local interference truth or clean samples given. |
| ICDM, IEEE JSAC 2026, DOI 10.1109/JSAC.2025.3643396 | Closest published interference-cancellation receiver (`ICDM_ORACLE`) | Existing reproduction of published 40-step sampler, original amplitude/table and true global SINR. Original oracle access is retained and disclosed. |
| SC-ICDM A4 | Proposed blind receiver | Same priors and 40-step sampler; observes received samples, thermal-noise power and block partition, not true SINR. |

Sources read: [SwinJSCC](https://arxiv.org/html/2308.09361),
[CDDM](https://arxiv.org/html/2309.08895),
[ICDM](https://arxiv.org/html/2505.19983v1),
[ISEC, AISTATS 2023](https://proceedings.mlr.press/v206/lee23c.html),
[ISEC full text](https://arxiv.org/html/2302.09174).

The DOI registry confirms CDDM: TWC 23(9), 11168–11183 (September 2024);
SwinJSCC: TCCN 11(1), 90–104 (February 2025); ICDM: JSAC 44, 2528–2543 (2026).
The existing codec was trained with image MSE, unlike the LPIPS-trained codec
in the ICDM paper. Its measured CBR is recorded from the actual tensors; it is
not assumed to equal the paper's 1/96. No published headline gain is claimed
to have been reproduced.

ISEC is a scientifically relevant next external reference: it refines the
codeword with an encoder–decoder cycle likelihood and a learned bias-free
denoiser. It is NOT equivalent to ordinary decode–encode iteration. Its denoiser
and original codec are not available as compatible project checkpoints, so it
is not included under a false ISEC label. A proper reproduction needs training
and validation of that complete estimator. The [authors' repository linked by
PMLR](https://github.com/changwoolee/isec-deep-jscc) provides pretrained
OpenImages models at CPP 1/6 and 1/16, not the current codec's 1/48. These are
real downloadable pretrained baselines, but using them unchanged would grant
three times (or more) the channel budget. Do not call them unavailable in
general or compare their scores as rate-matched evidence.

No FlowSem baseline is admitted: the available local README alone does not
establish peer-reviewed publication or comparable pretrained assets. BPG+LDPC
is also not replaced by an ideal-capacity surrogate under structured
interference. A physical digital baseline would need a specified modulation,
code, packet failure handling and identical channel-use accounting.

## CDDM adaptation disclosure (mandatory)

Original CDDM has three training stages, including decoder retraining after
inserting its independently trained U-Net denoiser (Sec. IV-B). This run does
not reproduce those stages. It asks a narrower question: does a Gaussian
denoising receiver based on the published update suffice with the same frozen
representation and prior as SC-ICDM? This controlled-backbone comparison has
a direct precedent: ICDM Sec. IV-C/Table II also replaces CDDM's U-Net with DiT
for fairness and chooses the starting step from SINR. That experiment reports
116 CDDM steps versus 40 ICDM steps. Our fixed 40-evaluation respacing is an
additional budget constraint, not that paper's convergence setting.

Implementation was checked against the authors' sampler at commit
`1d12e485e1062982fd87595fcf3f729f6d22b11e`, file
`CDDM/Diffusion/Diffusion.py` in
[the official repository](https://github.com/Wireless3C-SJTU/CDDM-channel-denoising-diffusion-model-for-semantic-communication).
The deterministic clean-estimate/re-noising equations match Algorithm 2.
Adaptations: DiT replaces U-Net; use the stored DiT's discrete training schedule;
match real noise variance Nc/2 and initialize sqrt(alpha_bar)*y; use up to 40
uniformly respaced indices instead of the original consecutive steps and
dataset-specific stopping cap. Freeze the same decoder for every method.
These are explicit design changes, so labels must say **CDDM receiver port**,
not “full CDDM reproduction” or “CDDM state of the art beaten”.

## Metrics and literature conventions

All metrics operate on the same clipped float RGB [0,1] image pair, without
resizing or 8-bit rounding. Save per-image scores before averaging.

| Metric | Direction and definition | Literature link |
|---|---|---|
| Image MSE | Lower; mean squared RGB error in [0,1] units | Pixel fidelity; distinguish from transmitted latent-signal MSE in CDDM/ICDM |
| PSNR | Higher; -10 log10(per-image MSE), then mean | SwinJSCC/CDDM/ISEC; not -10 log10(dataset mean MSE) |
| LPIPS-VGG v0.1 | Lower; official learned VGG distance, input 2x-1 | ICDM and ISEC use VGG; do not silently substitute AlexNet |
| MS-SSIM | Higher; four scales for 128x128; 11x11 Gaussian sigma1.5; normalized first four published weights | Four scales explicitly disclosed; not comparable with undeclared five-scale results |

Optional presentation transforms: MS-SSIM(dB)=-10 log10(1-MS-SSIM), as in
SwinJSCC/CDDM; LPIPS(dB)=10 log10(mean LPIPS), as in ICDM. Retain raw scores as
the main table so direction is unambiguous. Floors of 1e-12 only prevent infinite
logs on identical pairs. PSNR and image MSE are two summaries of the same error,
not independent evidence. Also record real-coordinate latent MSE after common
unit-complex-power projection; do not equate it to image MSE.

[Official LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity)
requires [-1,1] inputs when `normalize=False`.
[Wang et al., MS-SSIM](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)
combines each scale once. The local legacy expression multiplies the last-scale
SSIM term once for every earlier scale, changing its exponent. The loss class
returns 1-score and the evaluation wrapper converts it back; the issue is not
merely that wrapper's name. Preserve old files and evaluate all new methods with
one audited implementation. Verify its five-scale path against pytorch-msssim
and its four-scale path for identity and batch separation. Archive actual LPIPS
backbone/calibration hashes and all model/data/source hashes.

## Reporting gates

Execution note: another project's training and evaluation processes were
observed on the same host after this run started. They were not altered.
Quality evaluation continues with the locked settings, but receiver timings
must be treated as contention-affected descriptive records, not controlled
efficiency evidence. This annotation changes no method, data, or metric.

Do not merge smoke with validation. Require complete, unique paired records,
finite scores, metric unit tests and image identity tests. Report all five rows,
including adverse strata. Main comparison has two published algorithm families
plus a published codec architecture; it is not a broad benchmark across five
independent papers. Before a submission-level claim, add a complete additional
external-system reproduction (e.g. ISEC or full CDDM), freeze choices, and run a
genuinely independent test split. This validation alone cannot fill that gap.
