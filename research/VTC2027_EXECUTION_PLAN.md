# VTC 2027 execution plan: one claim, one mechanism, one confirmation

## Paper claim under test

For semantic transmission with unknown blockwise interference, a blind diffusion receiver can update the block amplitude from receiver-side conditional moments. The only proposed new mechanism is the observation-induced signal--interference cross covariance. The paper will not claim exact neural-posterior inference, universal fading robustness, or accurate physical power recovery.

## System model

For block \(b\),

\[
\mathbf y_b=\mathbf x_b+a_b\mathbf z_b+\mathbf n_b,
\qquad
\mathbf n_b\sim\mathcal{CN}(\mathbf0,N_0\mathbf I),\quad a_b\ge 0.
\]

The desired and interference latents have unit complex power over the complete frame. The receiver observes \(\mathbf y\), \(N_0\), the fixed block partition, two frozen diffusion priors and the decoder. It does not observe true SINR, \(a_b\), the profile label or either clean latent. Current claims are AWGN-only.

At reverse step \(t\), the frozen denoisers give projected point means \(d_x,d_z\). A declared diagonal Gaussian surrogate has real-coordinate variances \(v_x=v_z=v_t\). Conditioning it on \(y=x+a_{\rm old}z+n\) gives

\[
S=v_x+a_{\rm old}^2v_z+N_0/2,
\]
\[
\mu_x=d_x+\frac{v_x}{S}(y-d_x-a_{\rm old}d_z),\qquad
\mu_z=d_z+\frac{a_{\rm old}v_z}{S}(y-d_x-a_{\rm old}d_z),
\]
\[
C_{zz}=\frac{v_z(v_x+N_0/2)}{S},\qquad
C_{xz}=-\frac{a_{\rm old}v_xv_z}{S}.
\]

For fixed moments, the anchored amplitude update is the unique constrained minimizer

\[
a_b^*=\Pi_{[0,a_{\max}]}
\frac{\mu_{z,b}^{\mathsf T}(y_b-\mu_{x,b})-\operatorname{tr}C_{xz,b}
+\lambda_b a_b^{(0)}}
{\|\mu_{z,b}\|_2^2+\operatorname{tr}C_{zz,b}+\lambda_b}.
\]

Convex damping cannot increase this fixed-moment quadratic. It does not guarantee monotone PSNR or diffusion likelihood because the surrogate changes across diffusion steps.

## Minimum experiment set

The current validation contains six receivers only because each has a specific role:

1. `NO_ICDM`: direct-decoding boundary.
2. `A3`: one-shot blind block estimate.
3. `A4`: historical unregularized point update.
4. `A4_R`: point update with exactly the same stabilization as the proposed method.
5. `A5_DIAG`: conditioned means and variance without cross covariance.
6. `A5_FULL`: complete Gaussian-surrogate joint moments.

Validation uses 64 image pairs and two fixed, matched channel seeds. This is not a seed sweep. The evidence unit is an image pair; the primary result is the paired PSNR difference `A5_FULL - A4_R` with an image-cluster interval. Do not report a wall of mean±standard-deviation values. Runtime, cap/floor frequency and power error are diagnostics, not additional claims.

## Decision rule

- Continue with A5 only if `A5_FULL - A4_R` is positive with its image-cluster interval above zero, the mean severe-burst contrast is not worse, and there is no numerical failure.
- If A5 fails, do not tune it repeatedly. Retain the negative mechanism result and move to the simpler regularized point receiver or a receiver-observable bypass.
- A5_DIAG separates variance/conditioned-mean effects from the cross-covariance term. `A4_R` separates stabilization from uncertainty modeling.

## Independent confirmation

After validation, freeze one final receiver without changing its parameters. Confirm on 300 previously unused test image pairs (sorted indices 200--499), two fresh fixed channel seeds, and the same six SINRs and three profiles. Compare only direct decoding, A3, A4_R and the frozen finalist. Before launch, prove checkpoint identity and absence of selected validation/test image-hash overlap. This confirmation reports paired image-level effects, failure boundaries and controlled same-GPU runtime; it does not use test labels to select a gate.

## Five-page VTC story

1. Receiver information and block-interference system model.
2. Closed-form anchored joint-moment amplitude update and its limited fixed-moment descent guarantee.
3. Controlled point/diagonal/full ablation.
4. Independent paired confirmation and failure boundary.
5. Limitations: AWGN, known blocks, frozen priors, surrogate rather than calibrated neural uncertainty.

No retraining, architecture search, large seed sweep or extra robustness suite enters the main paper unless the primary mechanism survives confirmation.
