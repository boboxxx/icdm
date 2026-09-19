# Published-baseline validation artifact bundle

This directory contains the complete 2026-09-19 paired validation deliverable.
The run has 11,520 records: 64 image pairs, two channel seeds, six SINRs, three
interference profiles, and five receiver conditions. The audit in
`summary.json` passed complete-pairing, finite-value, PSNR/MSE consistency, and
smoke-repeat checks.

## Raw and derived results

- `records.csv`: raw, lossless CSV export of all 11,520 per-image receiver
  records. Mode-specific fields are blank when they do not apply.
- `records.jsonl`: original append-only raw records produced by the runner.
- `means.csv`, `by_sinr.csv`, `by_profile.csv`, and `strata.csv`: descriptive
  aggregations; no confidence intervals or standard-deviation summaries.
- `summary.json`: machine-readable audit, aggregates, paired mean differences,
  and favorable-pair fractions.
- `RESULTS.md`: compact result table and claim boundaries.
- `fig_metrics_overall.*` and `fig_four_metrics.*`: mean-only plots without
  error bars.

## Reconstructions and integrity

- `qualitative/`: 90 selected reconstructions and their 90 source images,
  covering stationary, alternating, and burst profiles at -7, 0, and 10 dB
  for all five receivers and two image indices.
- `RECONSTRUCTION_MANIFEST.csv`: one row per PNG with condition, sample index,
  dimensions, byte size, and SHA-256.
- `BUNDLE_MANIFEST.csv`: byte size and SHA-256 for every other file in this
  bundle.

## Reproducibility records

- `metadata.json`: frozen run manifest, metric definitions, image-pair hashes,
  model hashes, source hashes, perceptual-weight hashes, GPU, and package
  versions.
- `source_snapshot.tar.gz`: exact source snapshot archived by the run.
- `environment/`: Python/Conda dependency locks and host/runtime description
  from the actual CUDA execution environment.
- [`../PUBLISHED_BASELINE_PROTOCOL.md`](../PUBLISHED_BASELINE_PROTOCOL.md):
  preregistered comparison protocol, baseline provenance, adaptations, metric
  conventions, and claim limits.
- [`../../configs/published_baselines_v1.json`](../../configs/published_baselines_v1.json):
  run configuration.
- [`../../scripts/run_published_baselines.py`](../../scripts/run_published_baselines.py),
  [`../../scripts/analyze_published_baselines.py`](../../scripts/analyze_published_baselines.py),
  [`../../scripts/baseline_metrics.py`](../../scripts/baseline_metrics.py), and
  [`../../scripts/cddm_receiver.py`](../../scripts/cddm_receiver.py): execution,
  audit, metric, and receiver-port code.

Model checkpoints and the full CelebA/CIFAR-10 datasets are not redistributed.
Their cryptographic hashes are retained in `metadata.json`; access and licensing
remain the responsibility of the reproducer. The CDDM row is a receiver-only
shared-prior port, not a reproduction of the paper's complete three-stage
training pipeline.
