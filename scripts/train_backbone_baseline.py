"""Matched retraining for external DeepJSCC and MambaJSCC backbones.

The network definitions are imported from pinned external Git repositories.  This
adapter owns only the matched data split, complex channel convention, checkpoint
receipts, and resumable training loop used by the VTC 2027 comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def seed_all(value: int) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def dataset_digest(root: Path, samples) -> dict:
    """Content-address a split once; receipts can then detect data drift."""
    h = hashlib.sha256()
    total_bytes = 0
    for value, _ in sorted(samples):
        path = Path(value)
        size = path.stat().st_size
        total_bytes += size
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(bytes.fromhex(digest(path)))
    return {"sha256": h.hexdigest(), "files": len(samples), "bytes": total_bytes}


def make_loaders(args):
    if args.role == "source":
        train_root = args.data_root / "celeba" / "train"
        valid_root = args.data_root / "celeba" / "val"
        train_tf = transforms.Compose([
            transforms.RandomCrop((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        valid_tf = transforms.Compose([
            transforms.CenterCrop((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
    else:
        train_root = args.data_root / "cifar10" / "train"
        valid_root = args.data_root / "cifar10" / "val"
        train_tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        valid_tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    train = datasets.ImageFolder(train_root, transform=train_tf)
    valid = datasets.ImageFolder(valid_root, transform=valid_tf)
    if not train or not valid:
        raise ValueError("Empty training or validation split")
    inventories = {
        "train": dataset_digest(train_root, train.samples),
        "validation_full": dataset_digest(valid_root, valid.samples),
    }
    valid = Subset(valid, range(min(args.validation_images, len(valid))))
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                   generator=generator, **common),
        DataLoader(valid, batch_size=args.batch_size, shuffle=False, drop_last=False, **common),
        len(train), len(valid), inventories, generator,
    )


def mamba_config(root: Path, image_size: int, out_channels: int):
    sys.path.insert(0, str(root / "adaptive_selective_scan"))
    sys.path.insert(0, str(root))
    from configs.config import get_config

    class ConfigPaths:
        model_config_path = str(root / "configs/vssm/vssm_tiny_CelebA.yaml")
        train_config_path = str(root / "configs/train/vssm_tiny_CelebA.yaml")

    cfg = get_config(ConfigPaths)
    cfg.defrost()
    cfg.DATA.IMG_SIZE = image_size
    cfg.MODEL.VSSM.OUT_CHANS = out_channels
    cfg.CHANNEL.TYPE = "awgn"
    cfg.CHANNEL.SNR = [20]
    cfg.CHANNEL.ADAPTIVE = "ssm"
    cfg.TRAIN.LOSS = "MSE"
    cfg.TRAIN.USE_CHECKPOINT = False
    cfg.MODEL.DROP_PATH_RATE = 0.2
    cfg.freeze()
    return cfg


def build_models(args, device):
    root = args.external_root.resolve()
    if args.architecture == "deepjscc":
        sys.path.insert(0, str(root))
        from model import DeepJSCC
        # At 128x128, c=1 gives 2x32x32 real values = 1,024 complex uses.
        joint = DeepJSCC(c=1, channel_type="AWGN", snr=None)
        encoder, decoder = joint.encoder, joint.decoder
        effective = {"c": 1, "complex_pairing": "channel"}
    else:
        cfg = mamba_config(root, args.image_size, 32)
        from models.network import Mamba_encoder, Mamba_decoder
        encoder, decoder = Mamba_encoder(cfg), Mamba_decoder(cfg)
        binaries = sorted((root / "adaptive_selective_scan").glob("adaptive_selective_scan_cuda_core*.so"))
        if len(binaries) != 1:
            raise RuntimeError(f"Expected one compiled adaptive scan extension, found {binaries}")
        effective = {"out_channels": 32, "channel_adaptive": "ssm",
                     "complex_pairing": "height", "config_yaml": cfg.dump(),
                     "cuda_extension": str(binaries[0]),
                     "cuda_extension_sha256": digest(binaries[0])}
    return encoder.to(device), decoder.to(device), effective


def to_complex(latent: torch.Tensor, architecture: str) -> torch.Tensor:
    if architecture == "deepjscc":
        if latent.shape[1] % 2:
            raise ValueError("DeepJSCC channel dimension must be even")
        half = latent.shape[1] // 2
        return torch.complex(latent[:, :half], latent[:, half:])
    if latent.shape[2] % 2:
        raise ValueError("MambaJSCC height must be even")
    half = latent.shape[2] // 2
    return torch.complex(latent[:, :, :half], latent[:, :, half:])


def from_complex(latent: torch.Tensor, architecture: str) -> torch.Tensor:
    dim = 1 if architecture == "deepjscc" else 2
    return torch.cat((latent.real, latent.imag), dim=dim)


def roundtrip(encoder, decoder, images, architecture, snr, training=True):
    latent = encoder(images) if architecture == "deepjscc" else encoder(images, snr)
    # The public DeepJSCC reproduction squeezes its batch dimension when B=1.
    # Restore it in the adapter so batch-one confirmation remains independent.
    if architecture == "deepjscc" and latent.ndim == 3:
        latent = latent.unsqueeze(0)
    code = to_complex(latent, architecture)
    power = code.abs().square().flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(1e-12)
    code = code / power.sqrt()
    sigma = (1 / (2 * 10 ** (snr / 10))) ** 0.5
    received = code + sigma * (torch.randn_like(code.real) + 1j * torch.randn_like(code.real))
    # Match the established frozen-codec protocol: remove received global scale
    # before decoding, without using the clean latent or interference power.
    received = received / received.abs().square().flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(1e-12).sqrt()
    real = from_complex(received, architecture)
    reconstruction = decoder(real) if architecture == "deepjscc" else decoder(real, snr)
    return reconstruction, code


@torch.no_grad()
def validate(encoder, decoder, loader, args, device):
    encoder.eval(); decoder.eval()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    torch.manual_seed(args.seed + 900_000)
    torch.cuda.manual_seed_all(args.seed + 900_000)
    squared = 0.0
    pixels = 0
    uses = None
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            reconstruction, code = roundtrip(encoder, decoder, images, args.architecture, args.snr, training=False)
            squared += F.mse_loss(reconstruction.clamp(0, 1), images, reduction="sum").item()
            pixels += images.numel()
            uses = code[0].numel()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
    mse = squared / pixels
    return {"val_mse": mse, "val_psnr": -10 * np.log10(max(mse, 1e-15)),
            "complex_channel_uses": uses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["deepjscc", "mambajscc"], required=True)
    parser.add_argument("--role", choices=["source", "interference"], required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-images", type=int, default=1024)
    parser.add_argument("--snr", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Limit batches per epoch for an engineering smoke run only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.seed)
    train_loader, valid_loader, train_count, valid_count, inventories, loader_generator = make_loaders(args)
    encoder, decoder, effective = build_models(args, device)
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        _, code = roundtrip(encoder, decoder, probe, args.architecture, args.snr)
    uses = code[0].numel()
    expected_uses = args.image_size * args.image_size * 3 // 48
    if uses != expected_uses:
        raise ValueError(f"CBR mismatch: {uses} complex uses, expected {expected_uses}")
    lr = args.lr if args.lr is not None else (1e-3 if args.architecture == "deepjscc" else 1e-4)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    if args.architecture == "deepjscc":
        # Match the public reproduction's optimizer and schedule. Its archived
        # configurations use step_size=640, so no decay occurs in 40 epochs.
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=640, gamma=0.1)
        optimizer_name, scheduler_name = "Adam", "StepLR(step_size=640,gamma=0.1)"
    else:
        # Match the official MambaJSCC training loop.
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
        # Its 0.1-epoch GradualWarmupScheduler is stepped once per epoch: the
        # first step jumps to 2x, followed by T_max=epochs cosine steps. Express
        # that exact sequence with LambdaLR so resume state remains portable.
        def mamba_lr(completed_epochs: int) -> float:
            if completed_epochs == 0:
                return 1.0
            cosine_epoch = completed_epochs - 1
            return 1.0 + math.cos(math.pi * cosine_epoch / args.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, mamba_lr)
        optimizer_name = "AdamW"
        scheduler_name = "official GradualWarmup(2x,0.1 epoch)+CosineAnnealing"
    state_path = args.output / "latest.pt"
    best_path = args.output / "best.pt"
    log_path = args.output / "training.jsonl"
    start_epoch, best = 0, float("inf")
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["architecture"] != args.architecture or state["role"] != args.role:
            raise ValueError("Resume checkpoint identity mismatch")
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "loader_generator" in state:
            loader_generator.set_state(state["loader_generator"])
        start_epoch, best = state["epoch"], state["best"]
    receipt = {
        "status": "running", "architecture": args.architecture, "role": args.role,
        "external_repo": str(args.external_root), "external_commit": git_head(args.external_root),
        "data_root": str(args.data_root), "train_images": train_count,
        "data_inventories": inventories,
        "validation_images": valid_count, "image_size": args.image_size,
        "snr_db": args.snr, "complex_channel_uses": uses, "cbr": uses / probe[0].numel(),
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "learning_rate": lr, "effective_model": effective, "torch": torch.__version__,
        "optimizer": optimizer_name, "scheduler": scheduler_name,
        "adapter_sha256": digest(Path(__file__)),
        "gpu": torch.cuda.get_device_name(0), "started_at": time.time(),
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
        "smoke_batches": args.smoke_batches,
    }
    atomic_json(args.output / "receipt.json", receipt)
    mode = "a" if log_path.exists() else "x"
    try:
        with log_path.open(mode) as stream:
            for epoch in range(start_epoch, args.epochs):
                encoder.train(); decoder.train()
                total, seen = 0.0, 0
                began = time.perf_counter()
                for batch, (images, _) in enumerate(train_loader):
                    images = images.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    reconstruction, _ = roundtrip(encoder, decoder, images, args.architecture, args.snr)
                    loss = F.mse_loss(reconstruction, images)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
                    optimizer.step()
                    total += loss.item() * images.shape[0]
                    seen += images.shape[0]
                    if args.smoke_batches and batch + 1 >= args.smoke_batches:
                        break
                scheduler.step()
                metrics = validate(encoder, decoder, valid_loader, args, device)
                improved = metrics["val_mse"] < best
                best = min(best, metrics["val_mse"])
                state = {
                    "architecture": args.architecture, "role": args.role, "epoch": epoch + 1,
                    "best": best, "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "loader_generator": loader_generator.get_state(),
                    "effective_model": effective,
                }
                tmp = state_path.with_suffix(".tmp")
                torch.save(state, tmp); tmp.replace(state_path)
                if improved:
                    best_state = {k: state[k] for k in ["architecture", "role", "epoch", "best",
                                                           "encoder", "decoder", "effective_model"]}
                    tmp = best_path.with_suffix(".tmp")
                    torch.save(best_state, tmp); tmp.replace(best_path)
                row = {"epoch": epoch + 1, "train_mse": total / seen,
                       "seconds": time.perf_counter() - began, "lr": optimizer.param_groups[0]["lr"],
                       "best": best, **metrics, "time": time.time()}
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
        receipt.update(status="complete", completed_at=time.time(), best_val_mse=best,
                       best_sha256=digest(best_path), latest_sha256=digest(state_path))
    except Exception as error:
        receipt.update(status="failed", failed_at=time.time(), error=repr(error))
        raise
    finally:
        atomic_json(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
