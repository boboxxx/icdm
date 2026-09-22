# Matched DeepJSCC and MambaJSCC backbone protocol

This extension was requested after the frozen selector confirmation. It does not
change the selector, its thresholds, or any previously opened result. DeepJSCC
and MambaJSCC are codec-backbone references; they are not substitutes for a
published generative receiver baseline.

## Source and identity

- DeepJSCC architecture: `https://github.com/chunbaobao/Deep-JSCC-PyTorch`,
  pinned by commit and archived source hash in each run receipt. This is a public
  PyTorch reproduction because the original 2019 authors did not release a
  directly usable checkpoint for this protocol.
- MambaJSCC architecture: official
  `https://github.com/Wireless3C-SJTU/MambaJSCC`, likewise pinned by commit.
- Any compatibility/build patch must be saved as a diff next to the receipt.

## Matched retraining

Each architecture trains one source codec on the existing CelebA training split
and one interference codec on the existing CIFAR-10 training split. CIFAR-10 is
nearest-neighbour resized to 128 by 128, matching the frozen study. Validation
uses the first 1,024 items of the corresponding pre-existing validation split.
The loss is RGB MSE, channel is complex AWGN at 20 dB, and the budget is 40 full
epochs with the repository's architecture and a fixed seed. Checkpoint selection
uses validation MSE only. No confirmation image is used for training or model
selection.

Optimization follows the source repositories: DeepJSCC uses Adam at 0.001,
weight decay 0.0005, and the archived 640-epoch step interval (therefore no
decay within 40 epochs); MambaJSCC uses AdamW at 0.0001, weight decay 0.0001,
and its official two-times warm-up plus cosine schedule. The source codec uses
batch 256 for DeepJSCC and batch 32 for MambaJSCC after one-batch memory tests
on the 24-GB RTX 4090. These batch sizes also apply to the interference codec.

For 128 by 128 RGB images, every codec emits exactly 1,024 complex channel uses,
so CBR is 1/48. DeepJSCC uses channel-paired real feature maps; MambaJSCC uses the
official height pairing. The complex code is normalized to unit power, corrupted
at 20 dB, normalized at the receiver as in the frozen SwinJSCC training protocol,
and decoded by its own matched decoder.

## Confirmation

The best checkpoint from validation is evaluated on the already locked 256
CelebA/CIFAR-10 test pairs at offset 512, both channel seeds, six SINRs, and three
interference profiles. Each backbone's source and interference encoder create
its own matched semantic codewords. Channel seeds, block powers, metrics, and
image-pair clustering follow the frozen confirmation protocol. Results are
reported as `DeepJSCC (matched retraining, direct)` and
`MambaJSCC (matched retraining, direct)`.

The diffusion priors are specific to the SwinJSCC latent distribution. Gaussian,
joint, and selector receivers are therefore not attached to the two new codecs
without separately training new diffusion priors.
