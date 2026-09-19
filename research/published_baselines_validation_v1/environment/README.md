# Execution environment lock

This directory records the environment used for the completed 11,520-record
published-baseline validation run. It was verified on the execution host on
2026-09-19 against the environment path embedded in `metadata.json` by the
LPIPS weight record.

- Python executable: `/home/sheng/anaconda3/envs/cv/bin/python`
- Python: 3.10.15
- PyTorch: 2.5.0+cu118
- Torch CUDA runtime: 11.8
- cuDNN reported by PyTorch: 90100
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB
- NVIDIA driver: 591.86
- Host kernel: Linux 6.18.33.2-microsoft-standard-WSL2, x86_64
- Native `nvcc`: not present on the execution shell path; PyTorch used its
  packaged CUDA 11.8 runtime.

`requirements-frozen.txt` is the complete `pip freeze --all` output from the
actual `cv` environment. `conda-explicit.txt` locks the Conda-managed base of
that environment for `linux-64`; pip-installed packages, including CUDA-enabled
PyTorch, must additionally be installed from `requirements-frozen.txt`.

The model weights are intentionally not stored in Git. Their SHA-256 hashes,
along with image-pair, source, LPIPS calibration, and VGG weight hashes, are in
`../metadata.json`. Absolute paths in the metadata document the execution host;
they are not portable installation paths.
