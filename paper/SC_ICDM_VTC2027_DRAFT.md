# Self-Calibrated Diffusion Interference Cancellation under Nonstationary Interference

> **Draft status (not part of the manuscript).** This version is written around the completed A0--A4 evidence. The 200 image pairs were inspected during method development, so the reported results remain exploratory until the locked independent confirmation is complete. A5 has no image-reconstruction result and is therefore not presented as a contribution.

## Abstract

Interference cancellation diffusion models (ICDMs) can recover semantic features by combining separate diffusion priors for the desired signal and the interference. Existing ICDM inference, however, requires the receiver to know a single frame-level signal-to-interference-plus-noise ratio (SINR). That assumption is restrictive when the interference power is unknown and changes within a frame. We develop a SINR-blind receiver that estimates interference power from the received latent sequence and refines blockwise amplitudes during reverse diffusion. The refinement reuses the current denoised signal and interference estimates and therefore requires no additional neural-network evaluations. In experiments with stationary, alternating, and burst interference, blind global calibration remains within 0.008 dB of the true-global-SINR reference, while one-shot blockwise calibration gains 0.819 dB and 2.108 dB in the two nonstationary settings. Alternating calibration adds 0.599 dB over the one-shot blockwise receiver (95% image-pair-cluster bootstrap interval: 0.551--0.648 dB). The gain is not universal: alternating updates lose 0.109--0.174 dB under the strongest burst interference and remain 1.070 dB below direct decoding when interference is weak. These results show that reconstruction-optimal self-calibration need not recover the physical interference power and motivate reliability-gated activation rather than unconditional diffusion cancellation.

**Index Terms---** semantic communications, diffusion models, interference cancellation, blind calibration, nonstationary interference.

## I. Introduction

Diffusion priors provide a natural way to exploit the structure of transmitted semantic features at the receiver. In the recently proposed interference cancellation diffusion model (ICDM) [1], one diffusion model represents the desired latent distribution and a second model represents the interference distribution. A shared likelihood term couples the two reverse processes, allowing the receiver to reconstruct the desired feature while explaining structured interference. This design substantially improves reconstruction when the interference statistics used by the receiver match the channel.

One piece of side information remains easy to overlook: the published receiver selects the interference amplitude and guidance coefficients from the true frame-level SINR. In deployment, this quantity may be unavailable. More importantly, a single scalar cannot describe interference whose power changes within a frame. Replacing the true SINR with a global energy estimate addresses the first issue but not the second. Estimating each block independently captures local variation, but short-block estimates are noisy and are computed before the diffusion priors have separated the desired and interfering components.

This paper studies a narrower and more practical question: **can an ICDM receiver calibrate itself from the received latent sequence, without access to the true SINR, when interference power varies within a frame?** We answer this question with a receiver-side procedure that first obtains blind blockwise power estimates and then updates the associated amplitudes from the denoised signal and interference estimates produced during reverse diffusion. The update closes a feedback loop between reconstruction and calibration without retraining either prior or adding neural-network evaluations.

The contributions are threefold:

1. We formulate a SINR-blind ICDM protocol in which the receiver observes only the noisy latent sequence, the noise level, the desired-link channel state, and a fixed block partition. The simulated SINR, clean latent variables, and block powers are excluded from the proposed receiver.
2. We introduce blockwise alternating calibration inside reverse diffusion. The method uses the current denoised components to update local interference amplitudes and the corresponding guidance coefficients, thereby refining an otherwise one-shot energy estimate.
3. We isolate the value and the limits of self-calibration through matched comparisons. Blind global calibration preserves the mean performance of the true-global-SINR reference; local calibration improves nonstationary cases; and alternating feedback provides a further 0.599 dB gain over one-shot blockwise calibration. We also identify two failure regimes---severe burst interference and weak interference---in which unconditional updates or unconditional diffusion cancellation are undesirable.

The scope is deliberately limited. The experiments use AWGN, a known block partition, a known noise level, and fixed desired and interference priors. The proposed receiver is therefore SINR-blind, not fully blind. The evidence does not establish robustness to fading, unknown interference families, or unknown change points.

## II. Related Work

Diffusion models have been used as learned priors for wireless denoising and semantic reconstruction. CDDM applies a diffusion prior after channel equalization to suppress channel noise [2], whereas ICDM introduces separate signal and interference priors and couples their reverse processes through the observation model [1]. Our work retains the frozen ICDM priors and sampler. The distinction is receiver information: we remove the true-SINR input and allow interference power to vary across blocks.

Blind diffusion inverse problems also estimate unknown forward operators or nuisance parameters. GibbsDDRM, for example, alternates posterior sampling with inference of an unknown linear operator [3]. Moment-projected diffusion methods incorporate approximate conditional covariance information into inverse-problem guidance [4]. These works establish that parameter uncertainty and higher-order information are not new in diffusion inference. Our contribution is instead a lightweight calibration mechanism and a controlled receiver protocol for structured, intra-frame interference in semantic communication. The present method is a point-estimate update; it makes no posterior-calibration claim.

## III. System Model and Receiver Information

### A. Blockwise interference model

An image is encoded into \(N\) complex channel symbols \(\mathbf{x}=f_{\mathrm{enc}}(\mathbf{s})\), normalized such that

\[
\frac{1}{N}\lVert \mathbf{x}\rVert_2^2=1.
\]

An independently generated interference image is encoded into \(\mathbf{z}\) with the same framewise normalization. A fixed partition divides the transmitted order into disjoint blocks \(\{\mathcal B_b\}_{b=1}^{B}\). For AWGN, the observation in block \(b\) is

\[
\mathbf{y}_b=\mathbf{x}_b+a_b\mathbf{z}_b+\mathbf{n}_b,
\qquad
\mathbf{n}_b\sim\mathcal{CN}(\mathbf{0},N_0\mathbf{I}),
\tag{1}
\]

where \(a_b\geq0\) is an unknown interference amplitude. The nominal frame-level interference power is

\[
p_{\mathrm{int}}=\frac{1}{N}\sum_{b=1}^{B}|\mathcal B_b|a_b^2,
\tag{2}
\]

and the simulated SINR is \(10\log_{10}[1/(p_{\mathrm{int}}+N_0)]\). Because the latents are normalized over the complete frame rather than within each block, the realized block energy need not equal \(a_b^2\).

We evaluate three power profiles with identical nominal frame power. The stationary profile uses the same amplitude in every block. The alternating profile repeats normalized power multipliers \((0.25,1.75)\), and the burst profile repeats \((0,2)\). Real and imaginary components of one complex symbol share the same block coefficient.

### B. Receiver information

The receiver knows \(\mathbf{y}\), \(N_0\), the block partition, and the two frozen diffusion priors. It does not observe \(\mathbf{x}\), \(\mathbf{z}\), \(a_b\), the power-profile label, or the simulated SINR. The block partition is assumed known; detecting change points is outside the present scope. The true SINR is used only to generate and stratify test examples.

## IV. Self-Calibrated ICDM Receiver

### A. Blind power initialization

Under the unit-power and independence assumptions, the interference power in block \(b\) is initialized by subtracting the expected desired-signal and noise powers from the received energy:

\[
\widehat p_b^{(0)}=
\left[
\frac{1}{|\mathcal B_b|}\sum_{k\in\mathcal B_b}|y_k|^2
-1-N_0
\right]_+,
\qquad
\widehat a_b^{(0)}=\sqrt{\widehat p_b^{(0)}}.
\tag{3}
\]

The blind global receiver applies (3) to the complete frame; the blind blockwise receiver applies it separately to each block. Each estimated power is mapped to an estimated local SINR,

\[
\widehat\gamma_b=-10\log_{10}\!\left(\widehat p_b+N_0\right),
\tag{4}
\]

which selects the ICDM likelihood-guidance coefficients by linear interpolation of the original lookup table. Values outside the table are clipped to its endpoints. Neither (3) nor (4) accesses the true SINR.

### B. Diffusion-in-the-loop amplitude update

One-shot energy estimation ignores information revealed as the two diffusion priors separate the received mixture. We therefore update the block amplitudes during sampling. At reverse step \(t\), the pretrained noise predictors produce clean-point estimates

\[
\widehat{\mathbf{x}}_0^{(t)}=
\frac{\mathbf{x}_t-\sigma_t\boldsymbol\epsilon_{\theta_x}(\mathbf{x}_t,t)}{\alpha_t},
\qquad
\widehat{\mathbf{z}}_0^{(t)}=
\frac{\mathbf{z}_t-\sigma_t\boldsymbol\epsilon_{\theta_z}(\mathbf{z}_t,t)}{\alpha_t}.
\tag{5}
\]

We project each complete-frame estimate to unit complex power, consistent with the transmitter normalization, and denote the projected estimates by \(\widetilde{\mathbf{x}}_0^{(t)}\) and \(\widetilde{\mathbf{z}}_0^{(t)}\). Once \(\alpha_t\geq0.5\), the receiver updates each block using nonnegative least squares:

\[
\widehat a_b^{(t)}=
\left[
\frac{\operatorname{Re}\!\left\langle
\widetilde{\mathbf{z}}_{0,b}^{(t)},
\mathbf{y}_b-\widetilde{\mathbf{x}}_{0,b}^{(t)}
\right\rangle}
{\max\!\left(\left\lVert\widetilde{\mathbf{z}}_{0,b}^{(t)}\right\rVert_2^2,\tau_b\right)}
\right]_+,
\tag{6}
\]

where \(\tau_b=|\mathcal B_b|\epsilon_{\mathrm{machine}}\). If the denominator does not exceed \(\tau_b\), the preceding amplitude is retained. The updated power changes both the interference scale and the interpolated guidance coefficients at the next correction. With 40 order-3 UniPC steps, the schedule performs 21 updates.

Equation (6) reuses model outputs that ICDM already computes, so it introduces no additional denoiser call. It is nevertheless a heuristic point update: the framewise projection is not a posterior-moment identity, and \(\widehat a_b^{(t)}\) should not be interpreted as a calibrated measurement of physical interference power.

### C. Compared receivers

For clarity, the experiments separate four questions:

- **Direct:** equalization and JSCC decoding without diffusion cancellation.
- **Global oracle (A0):** the original receiver supplied with the true frame-level SINR.
- **Blind global (A2):** one frame-level estimate from (3).
- **Blind blockwise (A3):** one local estimate per block from (3), fixed during sampling.
- **Alternating calibration (A4):** A3 initialization followed by (6).

We also evaluate the power-convention correction A1 as an implementation control. A0 is a global-parameter reference, not a performance upper bound: it knows the correct frame average but not the nonstationary block powers.

## V. Experimental Setup

We use 200 CelebA test images as desired sources and pair them with CIFAR-10 interference images. The SwinJSCC encoder and decoder, the interference encoder, and both DiT priors are fixed for every receiver. Desired images are reconstructed at \(128\times128\) resolution. The AWGN SNR is 20 dB, and the simulated global SINR belongs to \(\{-7,-4,0,4,7,10\}\) dB. Each image pair is evaluated with three channel seeds, all three power profiles, and all six operating points. The latent block length is 64 complex symbols. Every diffusion receiver uses 40 UniPC steps with order three and identical model evaluations.

The baseline run contains 54,000 reconstructions across Direct and A0--A3; the matched A4 run adds 10,800 reconstructions. For each matched comparison, the image, interference image, channel realization, and diffusion initialization are identical. Checkpoint hashes and source hashes are recorded with the raw outputs.

We report PSNR and paired differences. Confidence intervals for A4 contrasts use 10,000 percentile bootstrap resamples of the 200 image-pair clusters, keeping every seed, SINR, and profile belonging to an image pair within the same resample. This cluster is the experimental unit; treating the 10,800 repeated records as independent would understate uncertainty. The current 200 pairs were inspected during method development, so all intervals in this draft are exploratory and not multiplicity-adjusted.

Receiver time includes calibration, diffusion sampling, output normalization, and JSCC decoding, with GPU synchronization. Data loading, source encoding, and channel simulation are excluded.

## VI. Results

### A. Removing true SINR does not reduce mean performance

The blind global receiver tracks the global oracle closely. Its mean difference from A0 is \(-0.005\) dB under stationary interference, \(-0.008\) dB under alternating interference, and \(-0.005\) dB under burst interference. These small descriptive gaps do not prove statistical equivalence, but they show that true receiver-side SINR is unnecessary for reproducing the mean performance of the published global receiver in this setup. Correcting the original amplitude convention changes mean PSNR by only 0.007--0.010 dB and therefore does not explain the gains below.

Local estimation matters when the interference is nonstationary. Relative to A0, the one-shot blockwise receiver loses 0.292 dB in the stationary case but gains 0.819 dB under alternating interference and 2.108 dB under burst interference. A correct global average is therefore insufficient when power variation within the frame is itself useful receiver information.

| Interference profile | Direct | Global oracle | Blind global | Blind blockwise | Alternating |
|---|---:|---:|---:|---:|---:|
| Stationary | 18.616 | 19.058 | 19.053 | 18.767 | **19.446** |
| Alternating | 18.811 | 18.939 | 18.931 | 19.758 | **20.352** |
| Burst | 19.047 | 18.903 | 18.898 | 21.012 | **21.535** |
| **All profiles** | 18.825 | 18.967 | 18.961 | 19.845 | **20.444** |

*Table I. Mean reconstruction PSNR (dB), averaged over images, seeds, and SINR values. Bold marks the largest mean in each row; the global oracle is not an upper bound under blockwise nonstationarity.*

### B. Alternating feedback improves one-shot calibration

Alternating calibration improves the one-shot blockwise receiver by 0.599 dB overall, with a 95% image-pair-cluster bootstrap interval of 0.551--0.648 dB. The mean gains are 0.680 dB under stationary interference, 0.594 dB under alternating interference, and 0.523 dB under burst interference; their exploratory cluster intervals are 0.629--0.729, 0.540--0.649, and 0.469--0.578 dB, respectively. A4 exceeds A3 in 79.1% of the 10,800 matched trials. Relative to A0, the alternating receiver gains 1.478 dB overall (95% cluster interval: 1.392--1.563 dB).

The improvement is strongest at moderate and weak interference. After pooling profiles, A4 gains 0.405 dB at 0 dB SINR, 0.952 dB at 4 dB, 1.122 dB at 7 dB, and 0.882 dB at 10 dB. Table II reveals an important exception: under burst interference, updating the point estimate is worse than retaining the A3 initialization at the two strongest interference levels.

| SINR (dB) | Stationary | Alternating | Burst | All profiles |
|---:|---:|---:|---:|---:|
| -7 | +0.323 | +0.206 | **-0.174** | +0.118 |
| -4 | +0.452 | +0.001 | **-0.109** | +0.115 |
| 0 | +0.315 | +0.454 | +0.446 | +0.405 |
| 4 | +1.084 | +0.925 | +0.846 | +0.952 |
| 7 | +1.050 | +1.109 | +1.208 | +1.122 |
| 10 | +0.854 | +0.870 | +0.923 | +0.882 |

*Table II. Paired PSNR difference A4 \(-\) A3 (dB). Negative values indicate that alternating updates reduce reconstruction quality.*

At burst SINR values of \(-7\) and \(-4\) dB, the losses are 0.174 dB (95% cluster interval: 0.126--0.224 dB) and 0.109 dB (0.025--0.197 dB), respectively. The result rules out the claim that diffusion-in-the-loop updates should always replace the initial local estimate.

### C. Better reconstruction does not require better power recovery

The reconstruction gain is not explained by more accurate physical power estimates. The A3 initializer has an aggregate block-power RMSE of 0.953, yet only 28.3% of A4 trials finish with a smaller per-trial power RMSE. At simulated SINR values of 0, 4, and 10 dB, the median final power estimates are 0.234, 0.024, and 0.004, whereas the corresponding nominal mean powers are 0.990, 0.388, and 0.090. Fourteen trials contain at least one estimated block power above 100; the maximum is 93,186.656.

These observations change the interpretation of the method. Equation (6) should not be described as recovering the true nuisance parameter. Instead, the alternating updates adapt the interference scale and likelihood guidance along the reconstruction trajectory. Scale ambiguity between \(\mathbf{x}\), \(\mathbf{z}\), and \(a_b\), together with low-energy interference estimates, may contribute to the discrepancy, but the historical logs do not record enough intermediate quantities to identify a single cause. The defensible conclusion is narrower and more useful: **reconstruction-optimal self-calibration need not coincide with physical interference-power estimation.**

### D. Runtime and the weak-interference boundary

The measured receiver times are 0.323 s per image for A3 and 0.347 s for A4 on an RTX 4090, a descriptive difference of 7.2%. Because these values come from separate batch-four runs, they should not be read as a controlled overhead measurement. The update itself adds no neural-network call; an interleaved same-device benchmark is required for a final runtime claim.

At 10 dB SINR, A4 is 0.882 dB better than A3 but remains 1.070 dB below direct decoding (95% cluster interval: \(-1.270\) to \(-0.867\) dB). When interference is already weak, running a 40-step cancellation process can therefore harm reconstruction. A deployable receiver should include a validation-selected bypass based on receiver-observable reliability, not on the unavailable true SINR.

## VII. Discussion and Limitations

The experiments separate three effects that are often conflated. First, true global SINR is not needed to match the mean performance of the original global receiver. Second, local power structure matters when interference is nonstationary. Third, feeding the evolving diffusion estimates back into calibration improves reconstruction beyond one-shot local initialization. The third effect survives all three profiles on average, but not every operating point.

The negative results are part of the contribution. Under severe burst interference, early point estimates are evidently not reliable enough to improve the block initializer. Under weak interference, diffusion cancellation itself is unnecessary. These two regimes suggest a three-way receiver decision among direct decoding, global calibration, and local iterative cancellation. Designing that decision from receiver observables is preferable to adding an oracle SINR threshold.

Several limitations bound the present claims. The channel is AWGN; the receiver knows the noise level and block partition; only one desired-image distribution, one interference-image distribution, one block length, and fixed matched priors are evaluated; and PSNR is the only reconstruction endpoint currently logged. LPIPS should be added to the independent confirmation because pixel fidelity alone does not establish perceptual improvement. Rayleigh fading, prior mismatch, randomized block boundaries, and other block lengths remain untested. Finally, the reported image pairs influenced method development. They are suitable for mechanism discovery but not for the final confirmatory claim.

## VIII. Conclusion

We removed true receiver-side SINR from diffusion interference cancellation and studied the harder case in which interference power changes within a frame. A blind global estimate preserved the mean performance of the true-global-SINR receiver, while blockwise initialization captured gains that a global parameter missed under alternating and burst interference. Closing the loop between diffusion reconstruction and local amplitude calibration added 0.599 dB over the one-shot blockwise receiver in the exploratory evaluation. The same experiments exposed two operating boundaries: point updates can hurt under severe burst interference, and diffusion cancellation can hurt when interference is weak. Moreover, the reconstruction gain does not imply accurate recovery of physical interference power. These findings support self-calibrated, reliability-gated interference cancellation rather than an unconditional receiver driven by oracle SINR.

## References

[1] T. Wu, Z. Chen, D. He, F. Yang, M. Tao, X. Xu, W. Zhang, and P. Zhang, "ICDM: Interference Cancellation Diffusion Models for Wireless Semantic Communications," *IEEE Journal on Selected Areas in Communications*, vol. 44, pp. 2528--2543, 2026, doi: 10.1109/JSAC.2025.3643396.

[2] T. Wu, Z. Chen, D. He, L. Qian, Y. Xu, M. Tao, and W. Zhang, "CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications," *IEEE Transactions on Wireless Communications*, vol. 23, no. 9, pp. 11168--11183, Sep. 2024, doi: 10.1109/TWC.2024.3379244.

[3] N. Murata, K. Saito, C.-H. Lai, Y. Takida, T. Uesaka, Y. Mitsufuji, and S. Ermon, "GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration," in *Proc. 40th Int. Conf. Machine Learning*, PMLR, vol. 202, pp. 25501--25522, 2023.

[4] B. Boys, M. Girolami, J. Pidstrigach, S. Reich, A. Mosca, and O. D. Akyildiz, "Tweedie Moment Projected Diffusions for Inverse Problems," *Transactions on Machine Learning Research*, 2024.

---

## Internal evidence gate before submission

This section is not part of the paper and must be removed before submission.

- Replace the exploratory A4 table and intervals with the frozen independent confirmation.
- Add LPIPS to the final evaluation; do not infer perceptual quality from PSNR.
- Keep A5 out of the title, abstract, and contribution list unless its locked validation and independent confirmation both support it.
- If A5 is neutral or negative, retain A4 as the proposed receiver and report the negative covariance result as an ablation only if space permits.
- Do not claim fully blind operation, calibrated power estimation, fading robustness, arbitrary interference robustness, or real-time execution.
- Benchmark A3 and A4 interleaved on the same GPU before stating a runtime percentage.
- Add the final venue-required disclosure of AI-assisted language editing in the acknowledgments.
