# Mathematical notes and implementation audit

Let q_b denote untruncated block mean excess energy and u_b=L_b/N. In an independent random-effects approximation, q_b=mu+delta_b+epsilon_b, Var(delta_b)=tau² and Var(epsilon_b)=s_b². Then

E[sum_b u_b(q_b-sum_j u_j q_j)²] = (1-sum_b u_b²)tau² + sum_b u_b(1-u_b)s_b².

The implemented weighted moment estimator follows by solving this equation and truncating tau² at zero. It handles unequal final block length. With equal blocks it becomes sample variance across q_b minus the mean s_b². For a single block there is no identifiable heterogeneity, so the implementation pools globally. Local powers are clipped only after heterogeneity is estimated to reduce clipping-induced distortion of the variance calculation.

The plug-in variance estimate assumes independent symbols within each block and independent block errors. Learned latents can violate both. Source-energy variation, source/interferer cross terms and correlated features can masquerade as interference nonstationarity. Accordingly this is approximate empirical shrinkage, not a calibrated posterior or a guaranteed stationary/nonstationary classifier. Subsequent validation should include source-only calibration or a correlation-aware variance estimate if needed.

The proposed trust update uses even complex coordinates to fit a nonnegative anchored ridge amplitude. A damped, clipped candidate is accepted only when it decreases residual on odd coordinates by the configured normalized margin. Zero/low interference-energy denominators reject the candidate. The denoisers still process all received coordinates, so the check is not statistically independent. It guards cross-coordinate inconsistency but cannot rule out overfitting or hallucination.

The comparator residual is evaluated at the candidate actually applied, rather than at an unconstrained LS optimum. This distinction matters: unconstrained in-sample LS can only improve its own residual and therefore a bare positive-improvement gate is weak evidence. The implementation uses a finite threshold, two subsets, finite denominators and a bounded step; no semantic-risk guarantee is claimed.

B3 keeps lambda and beta computed from initial hierarchical power at every update. B_GUIDE keeps physical amplitude fixed but updates guidance through the accepted controller amplitude. B_COUPLED updates both. These controls identify what changes the trajectory; a good reconstruction does not establish accurate physical power estimation.

Bypass really returns the equalized received feature without a prior call. All methods run one frame at a time with matched initialization seeds to avoid selection changing the RNG layout for surviving samples. Signal and interference network forward hooks measure NFE rather than inferring it from nominal solver steps. Cost of final JSCC decoding remains, even when diffusion NFE=0.

Phase-0 oracle routing uses per-trial reference-image PSNR and must never enter the deployment receiver or be called attainable performance. It establishes only available retrospective headroom among three existing methods.
