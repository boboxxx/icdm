# System model and mathematical scope

## 1. Transmission, block definition and information

An image s is encoded to N complex symbols x=f(s), with frame normalization ||x||²/N=1; an independently sourced interference image gives z with ||z||²/N=1. A known deterministic flattening and real/imag pairing define disjoint blocks B_b of sizes L_b (including a possible short tail). In the initial experiment:

    y_b = x_b + a_b z_b + n_b,  a_b >= 0,
    n ~ CN(0, N0 I),  N0 = 10^(-SNR/10).

This is AWGN with structured interference and an unknown block amplitude. The receiver has y, N0, the block partition, frozen desired/interference priors and decoder. It does not observe s,x,z,a,profile,SINR labels. Known receiver blocks are an assumption, not estimated change points. SNR/noise and interference distribution are known: do not claim fully blind or arbitrary unknown interference.

For a future known desired-link channel H, y=Hx+D(a)z+n; the interfering channel must either be explicitly modeled or absorbed consistently into the trained prior for z. Current AWGN evidence cannot support fading claims without retraining or matching priors.

Define nominal interference power p_nom=sum_b L_b a_b²/N and nominal SINR=1/(p_nom+N0). Measured frame interference power is p_real=||D(a)z||²/N and generally differs from p_nom because global unit-power normalization does not enforce unit power in every block. Log both. Burst duty and block location are simulator inputs, never receiver side information.

## 2. Exact conditional-moment amplitude identity

Use real stacking for derivation: r=(Re,Im), noise variance nu=N0/2 per real coordinate, and each complex block becomes 2L_b real coordinates. Let a fixed joint distribution q(x,z) have means mu_x,mu_z, covariances C_xx,C_zz,C_xz=E[(x-mu_x)(z-mu_z)^T]. For H=I and one block, minimizing

    J(a)=E_q ||y-x-a z||² + lambda (a-a_ref)²

over a>=0 gives

    a* = max(0, [mu_z^T(y-mu_x) - tr(C_xz) + lambda a_ref]
                 / [||mu_z||² + tr(C_zz) + lambda]).

For 0<=a<=a_max, clip this solution to that interval. If D=||mu_z||²+tr(C_zz)+lambda>0, J(a)=constant-2Na+Da² and this is its unique constrained minimum. Convex damping a_new=(1-eta)a_old+eta a*, eta in (0,1], does not increase J when a_old is feasible. This guarantee concerns a fixed q only; q changes during reverse diffusion, and neither PSNR nor the true posterior evidence is guaranteed to improve.

For general real H, replace the numerator term by mu_z^T(y-H mu_x)-tr(H C_xz). Off-block uncertainty can matter for non-diagonal operators; the implemented scalar conditioning assumes AWGN and diagonal prior surrogate.

Point estimation sets C_zz=C_xz=0. Keeping only C_zz is not the full moment update: the shared observation creates negative x-z cross covariance even if their prior surrogate is independent. A ridge penalty supplies an alternative stabilization and must be controlled in comparisons.

## 3. Computable Gaussian surrogate (approximation declared)

At a diffusion step, raw clean-point predictions are d_x=(x_t-sigma_t eps_x)/alpha_t and d_z similarly. The proposed implementation uses their framewise unit-complex-power projections to retain the A4 scale convention. This projection is a heuristic; it is not a posterior expectation under a sphere-constrained prior.

Declare the surrogate independently per real coordinate:

    q_t^0(x,z)=N(d_x,v_x) N(d_z,v_z),
    v_x=v_z=(tau² sigma_t²)/(alpha_t² tau²+sigma_t²), tau²=1/2.

The variance is the exact conditional variance for a scalar N(0,tau²) latent diffused with noise sigma_t, not for the learned non-Gaussian prior. Using neural means with this variance is a plug-in Gaussian approximation. It must never be labeled an exact neural posterior or a calibrated uncertainty without validation.

Condition this surrogate on the raw observation once at the current a_old:

    S = v_x + a_old² v_z + nu;
    r = y - d_x - a_old d_z;
    mu_x = d_x + v_x r/S;
    mu_z = d_z + a_old v_z r/S;
    C_zz = v_z - a_old² v_z²/S;
    C_xz = -a_old v_x v_z/S.

These formulas are exact for the declared surrogate. C_xx=v_x-v_x²/S; the full 2x2 covariance is PSD for positive variances/noise. Substitute the conditioned moments into Section 2, update a, and map a² to the same guidance table as A4. Rebuild q_t^0 each step; never multiply the same observation likelihood recursively as independent data. Reuse current model predictions, adding no neural calls in this design.

The update is a surrogate expected-residual step, not a convergence theorem for ICDM. A true EM monotonic-evidence claim would require an E-step under one fixed generative model; diffusion time and heuristic moment projection change that model.

## 4. Required comparisons and interpretation

A3 fixed initialization; A4 historical point update; A4_R point update with the same anchor/damping/cap as A5; A5_DIAG conditioned means and variance but cross term removed; A5_FULL full surrogate joint moments; NO_ICDM. Validate all on the same observations and random initial states. Future controls freeze the guidance coefficients while updating a, or freeze a while updating guidance, to distinguish cancellation strength from table adaptation.

Claim only reconstruction improvements measured on held-out data. Small denominator, excessive amplitude and uncertainty effects require diagnostics; the old final-power outliers alone do not prove their cause. A high-SINR gate must depend on receiver observables selected on validation, never test labels. A5 may fail; simpler regularized/gated A3/A4 remains a valid outcome.

## 5. Literature and venue anchors

- GibbsDDRM (Murata et al., ICML 2023): blind diffusion inverse problems with unknown operators already exist; no claim of first joint parameter/diffusion inference. https://proceedings.mlr.press/v202/murata23a.html
- Tweedie Moment Projected Diffusions (Boys et al.): higher-order diffusion conditional information already exists; novelty must be scoped to this receiver and validated block calibration. https://arxiv.org/abs/2310.06721
- VTC2027 Spring official CFP checked 2026-09-19: September 30, 2026 deadline, 5 pages without overlength, up to 7 pages with charges. https://events.vtsociety.org/vtc2027-spring/call-for-papers-2/

These literature anchors establish nearby approaches; they are not an exhaustive novelty search. A 5-page paper should focus on receiver information, one calibration rule, controlled evidence and failure boundaries.
