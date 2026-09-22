"""Small-data feasibility training for the original ICDM architecture.

Stages are resumable at epoch boundaries. Final inference weights contain the
raw codec state dicts and EMA diffusion state dicts expected by run_phase1.py.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Autoencoder.channel import Channel
from Autoencoder.SwinJSCC.encoder import create_SwinJSCC_encoder
from Autoencoder.SwinJSCC.decoder import create_SwinJSCC_decoder
from DiT import create_diffusion
from DiT.models import DiT_models


def config():
    from types import SimpleNamespace as NS
    return NS(
        MODEL=NS(
            MODEL_NAME="SwinJSCC", INF_MODEL_NAME="SwinJSCC", PATCH_SIZE=2,
            IN_CHANS=3, EMBED_DIMS=[128, 192, 256], DEPTHS=[2, 2, 6],
            NUM_HEADS=[4, 6, 8], WINDOW_SIZE=8, MLP_RATIO=4.0, OUT_CHANS=8,
        ),
        DATA=NS(IMG_SIZE=128, TEST_BATCH=1), CHANNEL=NS(TYPE="awgn"),
    )


def seed_all(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def atomic_save(obj, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def log(stream, **record):
    record["time"] = time.time()
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def loaders(root, batch_size, seed, interference=False):
    folder = "cifar10" if interference else "celeba"
    if interference:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop((128, 128)), transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop((128, 128)), transforms.ToTensor(),
        ])
    train = ImageFolder(root / folder / "train", transform=train_transform)
    val = ImageFolder(root / folder / "val", transform=val_transform)
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    return (
        DataLoader(train, shuffle=True, drop_last=True, generator=generator, **kwargs),
        DataLoader(val, shuffle=False, drop_last=False, **kwargs),
    )


def channel_roundtrip(model_encoder, model_decoder, images, channel, snr):
    latent = model_encoder(images)
    received, _, _ = channel.forward(latent, snr)
    received = received / received.abs().square().mean().clamp_min(1e-12).sqrt()
    stacked = torch.cat((received.real, received.imag), dim=2)
    return model_decoder(stacked)


@torch.no_grad()
def validate_codec(encoder, decoder, loader, channel, device, snr):
    encoder.eval(); decoder.eval()
    squared, pixels = 0.0, 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction = channel_roundtrip(encoder, decoder, images, channel, snr).clamp(0, 1)
        squared += F.mse_loss(reconstruction, images, reduction="sum").item()
        pixels += images.numel()
    mse = squared / pixels
    return {"val_mse": mse, "val_psnr": -10 * math.log10(max(mse, 1e-15))}


def resume_pair(encoder, decoder, optimizer, path, device):
    if not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"], state["best"]


def train_codec(stage, args, stream):
    is_interference = stage == "inf_codec"
    train_loader, val_loader = loaders(args.data_root, args.codec_batch, args.seed, is_interference)
    cfg = config(); channel = Channel(cfg)
    encoder = create_SwinJSCC_encoder(cfg).to(args.device)
    decoder = create_SwinJSCC_decoder(cfg).to(args.device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    state_path = args.output / "state" / f"{stage}.pt"
    start_epoch, best = resume_pair(encoder, decoder, optimizer, state_path, args.device)
    epochs = args.inf_codec_epochs if is_interference else args.codec_epochs
    log(stream, event="stage_start", stage=stage, start_epoch=start_epoch, epochs=epochs)
    for epoch in range(start_epoch, epochs):
        encoder.train(); decoder.train(); total = 0.0
        started = time.perf_counter()
        for images, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstruction = channel_roundtrip(encoder, decoder, images, channel, args.snr)
                loss = F.mse_loss(reconstruction, images, reduction="sum") / images.shape[0]
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics = validate_codec(encoder, decoder, val_loader, channel, args.device, args.snr)
        improved = metrics["val_mse"] < best
        best = min(best, metrics["val_mse"])
        state = {"epoch": epoch + 1, "best": best, "encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict()}
        atomic_save(state, state_path)
        if improved:
            atomic_save(encoder.state_dict(), args.output / "weights" / ("infencoder.pth" if is_interference else "encoder.pth"))
            atomic_save(decoder.state_dict(), args.output / "weights" / ("infdecoder.pth" if is_interference else "decoder.pth"))
        log(stream, event="epoch", stage=stage, epoch=epoch+1, train_loss=total/len(train_loader), seconds=time.perf_counter()-started, best=best, **metrics)
    del encoder, decoder, optimizer
    torch.cuda.empty_cache()


def update_ema(ema, model, updates, target_decay=0.9999):
    # A fixed 0.9999 decay leaves short feasibility runs dominated by the
    # randomly initialized model. Warm up the decay while retaining the
    # original long-run target.
    decay = min(target_decay, (1 + updates) / (10 + updates))
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1-decay)


def encode_latent(encoder, images, channel):
    with torch.no_grad():
        latent = encoder(images)
        latent, _ = channel.complex_normalize(latent, power=1)
    return latent


@torch.no_grad()
def validate_diffusion(model, encoder, loader, diffusion, channel, device, train_for):
    model.eval(); total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(20260915)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latent = encode_latent(encoder, images, channel)
        t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device, generator=generator)
        noise = torch.randn(latent.shape, device=device, generator=generator)
        loss = diffusion.training_losses(model, latent, t, "awgn", None, train_for=train_for, noise=noise)["loss"]
        total += loss.sum().item(); count += loss.numel()
    return total / count


def train_diffusion(stage, args, stream):
    is_interference = stage == "icdm_z"
    train_for = "inf" if is_interference else "s"
    train_loader, val_loader = loaders(args.data_root, args.dit_batch, args.seed, is_interference)
    cfg = config(); channel = Channel(cfg)
    encoder = create_SwinJSCC_encoder(cfg).to(args.device)
    encoder_path = args.output / "weights" / ("infencoder.pth" if is_interference else "encoder.pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device, weights_only=True), strict=True)
    encoder.eval().requires_grad_(False)
    model = DiT_models(in_channels=8, input_size=16).to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    diffusion = create_diffusion(timestep_respacing="")
    state_path = args.output / "state" / f"{stage}.pt"
    start_epoch, best, updates = 0, float("inf"), 0
    if state_path.exists():
        state = torch.load(state_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"], state["best"]
        updates = state.get("updates", start_epoch * len(train_loader))
    epochs = args.icdm_z_epochs if is_interference else args.icdm_s_epochs
    log(stream, event="stage_start", stage=stage, start_epoch=start_epoch, epochs=epochs)
    for epoch in range(start_epoch, epochs):
        model.train(); total = 0.0; started = time.perf_counter()
        for images, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            latent = encode_latent(encoder, images, channel)
            t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion.training_losses(model, latent, t, "awgn", None, train_for=train_for)["loss"].mean()
            loss.backward()
            optimizer.step()
            updates += 1
            update_ema(ema, model, updates)
            total += loss.item()
        val_loss = validate_diffusion(ema, encoder, val_loader, diffusion, channel, args.device, train_for)
        improved = val_loss < best; best = min(best, val_loss)
        checkpoint_due = (epoch + 1) % 5 == 0 or epoch + 1 == epochs
        if checkpoint_due:
            state = {"epoch": epoch+1, "best": best, "updates": updates, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict()}
            atomic_save(state, state_path)
            atomic_save(ema.state_dict(), args.output / "weights" / f"{stage}.pth")
            atomic_save(model.state_dict(), args.output / "weights" / f"{stage}_raw.pth")
        log(stream, event="epoch", stage=stage, epoch=epoch+1, train_loss=total/len(train_loader), val_loss=val_loss, best=best, seconds=time.perf_counter()-started)
    del model, ema, optimizer, encoder
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--snr", type=float, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--codec-batch", type=int, default=20)
    parser.add_argument("--dit-batch", type=int, default=20)
    parser.add_argument("--codec-epochs", type=int, default=40)
    parser.add_argument("--inf-codec-epochs", type=int, default=40)
    parser.add_argument("--icdm-s-epochs", type=int, default=50)
    parser.add_argument("--icdm-z-epochs", type=int, default=20)
    parser.add_argument("--stages", nargs="+", default=["inf_codec", "codec", "icdm_s", "icdm_z"])
    args = parser.parse_args()
    allowed = {"inf_codec", "codec", "icdm_s", "icdm_z"}
    if not set(args.stages) <= allowed:
        raise ValueError("Unknown stage")
    args.device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "weights").mkdir(exist_ok=True)
    (args.output / "state").mkdir(exist_ok=True)
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with (args.output / "training.jsonl").open("a") as stream:
        log(stream, event="run_start", stages=args.stages, seed=args.seed, torch=torch.__version__, gpu=torch.cuda.get_device_name(args.device))
        for stage in args.stages:
            if stage in {"codec", "inf_codec"}:
                train_codec(stage, args, stream)
            else:
                train_diffusion(stage, args, stream)
        log(stream, event="run_complete", stages=args.stages)


if __name__ == "__main__":
    main()
