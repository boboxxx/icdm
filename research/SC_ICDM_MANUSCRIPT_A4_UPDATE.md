# Manuscript update: A4 alternating point calibration

This file contains a working English draft, corrected by the 2026-09-19 evidence audit; it is not submission-ready. Historical test images were inspected during development and are exploratory. New A5 validation is pending, with no claimed gain. See SYSTEM_MODEL_VTC2027.md for the declared Gaussian-surrogate extension. Citation keys, equation numbers, table numbers, and cross-references should be aligned with the final LaTeX source when it becomes available.

## Terminology ledger

| Canonical term | Definition and use |
|---|---|
| ICDM | interference cancellation diffusion model; use for the published receiver family |
| global-parameter oracle | A0; receives the true global SINR, but is not a mathematical performance upper bound |
| blind global ICDM | A2; estimates one interference power per frame |
| blind blockwise ICDM | A3; estimates one power per block before diffusion sampling |
| alternating point calibration | A4; updates block amplitudes from diffusion point estimates during sampling |
| nonstationary interference | interference whose power varies within a frame; instantiated by alternating and burst profiles |
| direct decoding | `NO_ICDM`; equalization and JSCC decoding without diffusion interference cancellation |

## Revised title

**Self-Calibrated Diffusion Interference Cancellation under Nonstationary Interference**

## Revised abstract

Diffusion-based interference cancellation for semantic communications commonly assumes that the receiver knows a global signal-to-interference-plus-noise ratio (SINR), although this scalar may be unavailable and can be insufficient when interference power changes within a frame. We study blind self-calibration of an interference cancellation diffusion model (ICDM) without providing the receiver with the true SINR. A global energy estimator reproduces the true-global-SINR reference within 0.008 dB on average, whereas blockwise initialization improves reconstruction by 0.819 dB under alternating interference and 2.108 dB under burst interference. We further introduce alternating point calibration, which updates block amplitudes from the current denoised signal and interference estimates inside the reverse diffusion process. Across 200 historically evaluated image pairs, three channel seeds, six SINR values, and three interference profiles, the alternating receiver improves the blind blockwise initializer by 0.599 dB (image-pair-cluster bootstrap 95% confidence interval, 0.551–0.648 dB). The benefit is conditional: point updates lose 0.109–0.174 dB under the strongest burst settings, remain 1.070 dB below direct decoding at 10 dB SINR, and do not consistently improve physical power estimation. These exploratory results motivate independent confirmation, equally regularized moment-update controls, and validation of a receiver bypass; they do not establish the benefit of uncertainty modeling.

## Introduction: revised contribution paragraph

This work asks whether diffusion interference cancellation requires true global SINR and whether a single frame-level interference parameter remains adequate when interference power changes within a transmission. We make three contributions. First, we formulate a receiver-only calibration protocol in which the target methods observe the received latent signal, desired-link channel state information, and noise level, but not the simulated SINR, clean latent variables, or true block powers. Second, we introduce blind global and blockwise initializers and an alternating point-calibration receiver that feeds denoised signal and interference estimates back into block-amplitude and guidance updates during reverse diffusion. Third, we evaluate these receivers using 64,800 reconstruction records with matched images, interference images, channel noise, and diffusion initialization. The comparisons examine four implementation changes: correcting the published power convention, removing true global SINR, representing intra-frame nonstationarity, and coupling calibration to diffusion reconstruction. The results show conditional improvements in this exploratory sample, with extreme power estimates in some trials and losses relative to direct decoding under weak interference. Amplitude changes and guidance changes have not yet been causally separated.

## Method

### Problem formulation

For block \(b\), we write the complex latent observation as

\[
\mathbf y_b=\mathbf h_b\odot\mathbf x_b+a_b\mathbf z_b+\mathbf n_b,
\]

where \(\mathbf x_b\) and \(\mathbf z_b\) are the desired and interference latents, \(\mathbf h_b\) is the desired-link channel coefficient, \(a_b\geq0\) is an unknown interference amplitude, and \(\mathbf n_b\) is additive noise. The receiver knows the desired-link channel state and noise level. The true global SINR and the simulated values of \(a_b\) are used only to generate and stratify test examples; they are not inputs to A2–A4.

The evaluated nonstationary profiles preserve the same length-weighted nominal amplitude power while redistributing it across blocks. Both transmitted latents have unit complex power per frame, not per block; realized interference energy also depends on each block's latent energy and need not equal the nominal power. The stationary profile assigns equal power to every block. The alternating profile repeats normalized multipliers \((0.25,1.75)\), and the burst profile repeats \((0,2)\). Real and imaginary components share a block coefficient, and blocks follow the transmitted latent-symbol order.

### Blind initialization

For AWGN and unit-complex-power signal and interference latents, A3 initializes the power in block \(b\) as

\[
\widehat p_b^{(0)}=
\left[
\frac{1}{|\mathcal B_b|}\sum_{k\in\mathcal B_b}|y_k|^2
-1-\sigma_n^2
\right]_+,
\qquad
\widehat a_b^{(0)}=\sqrt{\widehat p_b^{(0)}}.
\]

A2 uses the same estimator over the complete frame. The estimated local SINR is

\[
\widehat\gamma_b=-10\log_{10}\left(\widehat p_b+10^{-\mathrm{SNR}/10}\right),
\]

from which the guidance coefficients are obtained by linear interpolation of the published ICDM guidance table. This interpolation is clipped to the table endpoints and does not use the true test SINR.

### Alternating point calibration

A4 begins from the A3 block estimates. At reverse-diffusion time \(t\), the two pretrained noise predictors give denoised point estimates

\[
\widehat{\mathbf x}_0^{(t)}=
\frac{\mathbf x_t-\sigma_t\boldsymbol\epsilon_{\theta_x}(\mathbf x_t,t)}{\alpha_t},
\qquad
\widehat{\mathbf z}_0^{(t)}=
\frac{\mathbf z_t-\sigma_t\boldsymbol\epsilon_{\theta_z}(\mathbf z_t,t)}{\alpha_t}.
\]

To enforce the transmitter's framewise scale convention, each denoiser point estimate is projected to unit complex power over the whole frame, never independently per block. This heuristic projection changes the denoiser estimate and is not a posterior-moment identity. Once \(\alpha_t\geq0.5\), A4 updates each block by nonnegative least squares,

\[
\widehat a_b^{(t)}=
\left[
\frac{\operatorname{Re}\left\langle
\widetilde{\mathbf z}_{0,b}^{(t)},
\mathbf y_b-\widetilde{\mathbf x}_{0,b}^{(t)}
\right\rangle}
{\max\!\left(\left\|\widetilde{\mathbf z}_{0,b}^{(t)}\right\|_2^2,\tau_b\right)}
\right]_+.
\]

Here \(\tau_b=|\mathcal B_b|\,\epsilon_{\rm machine}\). When the denominator is no greater than this threshold, the implementation retains the preceding amplitude rather than using the displayed quotient. There is no additive ridge in historical A4. The resulting power, local SINR, and guidance coefficients replace their previous values for the next correction. With the 40-step order-3 UniPC sampler, this schedule performs 21 alternating updates. A4 does not use posterior variances or signal–interference cross-covariances; those quantities are reserved for the uncertainty-aware extension.

## Experimental setup

We evaluated the receivers on 200 CelebA test-split images paired with CIFAR-10 interference images. These images were inspected during development, so the present analysis is exploratory rather than independent confirmation. The desired and interference encoders and the two diffusion priors were fixed across methods. The channel SNR was 20 dB, and the global SINR used for simulation was selected from \(\{-7,-4,0,4,7,10\}\) dB. Each image was evaluated with three channel seeds under stationary, alternating, and burst interference, producing 10,800 matched trials per receiver. The transmitted complex latent contained 64-symbol calibration blocks. All diffusion receivers used 40 UniPC steps, order three, and identical model evaluations. For A4 contrasts we report reconstruction PSNR and percentile 95% confidence intervals from 10,000 bootstrap resamples of the 200 image-pair clusters (seed 20260919), preserving seeds, SINRs, and profiles within each cluster. Stratum intervals are exploratory and not multiplicity-adjusted. Earlier record-level normal intervals are withdrawn; A2−A0 and A3−A0 are reported descriptively pending their corresponding cluster audit. Receiver time includes calibration, diffusion sampling, output normalization, and JSCC decoding, with GPU synchronization around each measured interval.

The comparison contains direct decoding without ICDM; A0, which receives the true global SINR and retains the original amplitude convention; A1, which corrects the amplitude convention; A2, which estimates a single frame-level power; A3, which initializes independent block powers; and A4, which alternates between diffusion reconstruction and block-amplitude updates. We refer to A0 as a global-parameter oracle rather than a performance upper bound because it does not represent intra-frame power variation.

## Results

### Local calibration outperforms a true global parameter under nonstationarity

The blind global receiver A2 reproduced A0 within -0.005, -0.008, and -0.005 dB under stationary, alternating, and burst interference, respectively. Similar mean values do not establish statistical equivalence. Correcting the amplitude convention in A1 changed mean PSNR by only 0.007–0.010 dB. These descriptive controls indicate that these two changes have small mean effects in this implementation.

In contrast, A3 exceeded A0 by 0.819 dB under alternating interference and by 2.108 dB under burst interference. A3 lost 0.292 dB under stationary interference, indicating that local estimation is useful only when block structure carries information that compensates for its additional estimation variance.

### Alternating feedback improves one-shot blockwise calibration

| Interference profile | Direct | A0 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|---:|
| Stationary | 18.616 | 19.058 | 19.053 | 18.766 | **19.446** |
| Alternating | 18.811 | 18.939 | 18.931 | 19.758 | **20.352** |
| Burst | 19.046 | 18.904 | 18.899 | 21.012 | **21.535** |
| All profiles | 18.825 | 18.967 | 18.961 | 19.845 | **20.444** |

A4 improved A3 by 0.599 dB overall (image-cluster 95% CI, 0.551–0.648 dB), with profile-level gains of 0.680 dB under stationary, 0.594 dB under alternating, and 0.523 dB under burst interference. It exceeded A3 in 79.1% of the 10,800 matched trials. Relative to A0, A4 gained 0.388, 1.413, and 2.632 dB under the three profiles. These results support a reconstruction gain for the coupled receiver in this sample; they do not isolate the mechanism of that gain.

The benefit was not uniform. A4 lost 0.174 dB (-0.224 to -0.126 dB) relative to A3 for burst interference at -7 dB and lost 0.109 dB (-0.197 to -0.025 dB) at -4 dB. Its largest gains occurred at 4–10 dB, where it improved A3 by 0.846–1.208 dB depending on the profile.

| SINR (dB) | Stationary | Alternating | Burst |
|---:|---:|---:|---:|
| -7 | +0.323 | +0.206 | -0.174 |
| -4 | +0.452 | +0.001 | -0.109 |
| 0 | +0.315 | +0.454 | +0.446 |
| 4 | +1.084 | +0.925 | +0.846 |
| 7 | +1.050 | +1.109 | +1.208 |
| 10 | +0.854 | +0.870 | +0.923 |

### Reconstruction gains do not imply calibrated physical power

The A3 initializer had an aggregate block-power RMSE of 0.953. Only 28.3% of A4 trials ended with a smaller per-trial power RMSE, and A4 increasingly underestimated interference as SINR increased. The median final estimates were 0.234, 0.024, and 0.004 at 0, 4, and 10 dB, compared with true mean powers of 0.990, 0.388, and 0.090. Fourteen trials contained at least one estimate above 100, with a maximum of 93,186.656. The historical logs did not record update denominators; a small-denominator explanation is a hypothesis, not a measured cause. Amplitude-only and guidance-only ablations are needed before attributing the reconstruction gain to either mechanism.

### Runtime and bypass boundary

A4 required 0.347 s per image, compared with 0.323 s for A3, a descriptive 7.2% difference between separate batch-four runs on an RTX 4090. This is not a controlled overhead estimate; the planned comparison interleaves methods on the same GPU. At 10 dB, A4 improved A3 by 0.882 dB when profiles were pooled, but it remained 1.070 dB below direct decoding. This result motivates testing a blind, validation-selected bypass; no such gate has yet been validated.

## Discussion

The exploratory experiments show that the blind global receiver has similar average PSNR to the global-parameter oracle and that local calibration improves the evaluated nonstationary cases. A4 improves A3 under each profile on average, but the gain does not establish accurate physical-power recovery or identify the role of guidance adaptation. Runtime needs a controlled same-device comparison.

The point estimates should not be interpreted as calibrated measurements of interference power. Signal–interference ambiguity, small denominators and guidance adaptation are possible explanations, but historical final-state logs cannot distinguish them. The locked extension compares equally stabilized point, diagonal-moment and full cross-covariance updates. Its moments are exact only for a declared local Gaussian surrogate; neither true neural-posterior accuracy nor monotone reconstruction quality is guaranteed. Variance regularization must earn its place through ablations rather than being assumed to solve the observed failures.

The method also has a clear operating boundary. Under severe burst interference at -7 and -4 dB, preserving the A3 initialization was slightly better than applying point updates. Under weak interference at 10 dB, every diffusion receiver remained inferior to direct decoding. A practical receiver should therefore estimate both nonstationarity and intervention reliability, selecting among direct decoding, global calibration, and local iterative cancellation. The present evidence is restricted to AWGN, one desired-data distribution, one interference-data distribution, one latent block size, and fixed pretrained priors; Rayleigh fading, prior mismatch, other block lengths, and uncertainty-aware updates remain to be evaluated.

## Revised conclusion

We investigated self-calibrated diffusion interference cancellation without true receiver-side SINR. A blind global estimate matched the true-global-SINR reference, while blockwise initialization exposed the insufficiency of a single global parameter under nonstationary interference. Alternating point calibration improved the blockwise receiver by 0.599 dB overall (image-cluster 95% CI, 0.551–0.648 dB) in the historical exploratory sample. The gain was conditional: point updates degraded the strongest burst cases, produced unreliable physical power estimates, and did not surpass direct decoding at 10 dB. These boundaries motivate an uncertainty-aware calibration rule and a reliability-gated receiver rather than unconditional blockwise cancellation.

## Claim–evidence map for author review

| Claim | Evidence | Status |
|---|---|---|
| Blind global calibration has similar mean PSNR in this setup | A2 − A0 means between −0.008 and −0.005 dB | Descriptive; no formal equivalence test |
| Global SINR is insufficient under nonstationary interference | A3 − A0 = +0.819 dB alternating and +2.108 dB burst | Supported |
| Iterative coupling adds value beyond blockwise initialization | A4 − A3 = +0.599 dB [0.551, 0.648] | Supported in the historical exploratory sample |
| A4 estimates physical block power more accurately | Only 28.3% improve power RMSE; severe outliers occur | Not supported; explicitly rejected |
| A4 should always replace A3 | Negative results for strong burst interference | Not supported |
| ICDM should always run | A4 is 1.070 dB below direct decoding at 10 dB | Not supported; gate remains to be validated |
| Findings generalize to fading and unseen interference priors | Not evaluated here | Needs evidence |

## Author inputs still needed

- Insert the canonical citation key for the published ICDM paper.
- Confirm the final venue template, page limit, and whether confidence intervals belong in the main table or appendix.
- Decide whether the final submission stops at A4 or presents A4 only as an ablation for A5/A6.
- Add Rayleigh and prior-mismatch claims only after matching experiments exist.
