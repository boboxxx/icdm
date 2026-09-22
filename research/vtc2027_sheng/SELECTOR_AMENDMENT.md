# Development design correction, 2026-09-21

Mathematical review identified an unsupported assignment assumption: the initial family allowed Gaussian reception only below a power threshold. Disturbance-model selection should test both directions. Add a single binary direction shared across the two heterogeneity regions. No extra feature, receiver action or confirmation feedback is introduced.

Timing: engineering smoke outcomes were inspected; formal development quality outcomes had not been analyzed or inspected. Only progress counters from the partial initial development were read. Confirmation and pressure stages had not started. The partial original development is preserved as `results/development_initial_direction_excluded` and excluded from fitting and the manuscript. Restart all 64 development images under the revised protocol/source signature. The same declared channel/sampler seeds are retained. Initial protocol is archived separately.

Candidates: 26 energy-only, 2,028 energy-plus-heterogeneity. Selection still uses eight image-group folds, mean PSNR, the original tie rules and fixed-receiver fallback. All thresholds and direction are frozen before independent confirmation. This is a development design correction, not a confirmatory hypothesis changed after its outcomes.
