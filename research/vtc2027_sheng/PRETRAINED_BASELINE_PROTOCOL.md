# Public-pretrained diagnostic protocol

This track is additional to the matched-retraining comparison. It never changes
the frozen selector, matched training, or confirmation data. Its purpose is to
show what the public weights do when transferred without fine-tuning.

DeepJSCC uses the repository-bundled ImageNet checkpoint from historical source
commit `e1cef6aeadedde9ff359760936a2ddf2540dd821`. It was trained at SNR 19 with
nominal real ratio 0.33. At 128 by 128 it emits 15,979 real values, padded by one
zero for 7,990 complex uses (effective complex CBR 0.16256), so it is a
high-bandwidth, off-protocol diagnostic.

MambaJSCC uses two official public checkpoint pairs: CLIC2021/AWGN/SNR 10 with
no channel adaptation, and DIV2K/Rayleigh/multi-SNR with CSI-ReST evaluated at
its supported SNR 20 state. Both were trained at 256 by 256; their fully
convolutional models are transferred to the locked 128 by 128 inputs without
fine-tuning. They emit 1,024 complex uses (CBR 1/48). Any legacy `timm`
deserialization shim and both repository-provided CUDA extensions are hashed.

For all three diagnostics, the same public encoder processes desired CelebA and
nearest-neighbour-upsampled CIFAR-10 images. This represents an off-the-shelf
single-codec deployment; no CIFAR-10-specific interference encoder is trained.
The official latent power normalization and decoder rescaling are retained.
The locked image pairs, channel seeds, 18 conditions, block size, and batch-one
metrics are reused. Results remain separate from the matched-retrained rows and
must be labelled `public pretrained; protocol mismatch`.
