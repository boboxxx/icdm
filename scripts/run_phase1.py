"""A0–A3 paired evaluation using explicit local checkpoints and image folders.

Run from repository root: python -m scripts.run_phase1 --manifest ... --output ...
No downloads or training. --preflight validates inputs without loading models.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import torch
from PIL import Image
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Autoencoder.channel import Channel
from Diffusion import ICDMSampler
from Diffusion.calibration import equalize, guidance_parameters, to_complex


def seed(value):
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def files(path):
    return sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def load_image(path, size, device):
    # Explicit center crop; never resize test images silently.
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < size:
            raise ValueError(f"Image smaller than configured crop {size}: {path}")
        left, top = (width-size)//2, (height-size)//2
        array = np.array(image.crop((left, top, left+size, top+size)), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255


def load_weights(model, spec, prefix=""):
    state = torch.load(spec["path"], map_location="cpu", weights_only=True)
    if spec.get("state_key"):
        state = state[spec["state_key"]]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain an explicit state dict")
    if prefix and state and all(key.startswith(prefix) for key in state):
        state = {key[len(prefix):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def make_config(manifest):
    with open(manifest["model_config"]) as stream:
        model = yaml.safe_load(stream)["MODEL"]
    model["OUT_CHANS"] = manifest["out_channels"]
    return NS(MODEL=NS(**model), DATA=NS(IMG_SIZE=manifest["image_size"], TEST_BATCH=1), CHANNEL=NS(TYPE=manifest["channel_type"]))


def codec(architecture, component, config, spec):
    if architecture not in {"SwinJSCC", "DeepJSCC", "MambaJSCC"}:
        raise ValueError("Unsupported codec architecture")
    module = importlib.import_module(f"Autoencoder.{architecture}.{component}")
    model = getattr(module, f"create_{architecture}_{component}")(config)
    return load_weights(model, spec, component + ".")


def oracle_parameters(sinr, channel_type, device):
    # Preserve the published integer-key lookup, including its supported grid.
    key = int(sinr)
    supported = {-7, -4, 0, 4, 7, 10} | ({20} if channel_type == "awgn" else set())
    if key not in supported:
        raise ValueError(f"A0 table has no SINR key {key}; select a published grid point")
    # Python scalars preserve the exact upstream constants (no float32 roundtrip).
    if channel_type == "awgn":
        lam = {-7:.3, -4:1, 0:1.4, 4:3, 7:4.5, 10:5.5, 20:3}
        beta = {-7:.1, -4:.7, 0:1, 4:1, 7:1, 10:.5, 20:2}
    else:
        lam = {-7:.3, -4:.5, 0:1, 4:1.8, 7:2.2, 10:2.5}
        beta = {-7:.1, -4:.1, 0:.3, 4:1, 7:1, 10:1}
    return lam[key], beta[key]


def block_powers(count, block_size, snr, sinr, profile, device):
    number = (count + block_size - 1) // block_size
    pattern = {"stationary": [1.], "alternating": [.25, 1.75], "burst": [0., 2.]}[profile]
    if profile != "stationary" and number < 2:
        raise ValueError("Nonstationary profiles need at least two blocks")
    values = torch.tensor([pattern[i % len(pattern)] for i in range(number)], device=device)
    sizes = torch.tensor([min(block_size, count-i*block_size) for i in range(number)], device=device)
    values = values / ((values * sizes).sum() / count)
    return (values * (10 ** (-sinr/10) - 10 ** (-snr/10))).unsqueeze(0)


@torch.no_grad()
def recover(mode, sampler, decoder, received, h, snr, sinr, channel_type, block_size, device, a4_min_alpha=.5):
    sync(device)
    start = time.perf_counter()
    estimates = None
    if mode == "NO_ICDM":
        feature = equalize(received, h, snr, channel_type)
    elif mode in {"A0", "A1"}:
        lam, beta = oracle_parameters(sinr, channel_type, device)
        feature, _ = sampler.SIC_sampling(snr, sinr, equalize(received, h, snr, channel_type), h, Lambda=lam, Beta=beta, amplitude_mode="original" if mode == "A0" else "power_consistent")
    elif mode in {"A2", "A3"}:
        # The receiver method has no true SINR, power, clean signal or interference arguments.
        feature, _, estimates = sampler.SIC_sampling_estimated(snr, received, h, channel_type, None if mode == "A2" else block_size)
    elif mode == "A4":
        # A4 receives exactly the same blind inputs as A3 and alternates only
        # after the reverse process has produced sufficiently denoised points.
        feature, _, estimates = sampler.SIC_sampling_alternating(
            snr, received, h, channel_type, block_size, min_alpha=a4_min_alpha
        )
    else:
        raise ValueError(f"Unknown receiver mode: {mode}")
    sync(device)
    sampled = time.perf_counter()
    # Keep evaluation semantics independent of batch size.
    dims = tuple(range(1, feature.ndim))
    scale = (2 * feature.square().mean(dim=dims, keepdim=True)).clamp_min(1e-12).sqrt()
    feature = feature / scale
    reconstruction = decoder(feature)
    sync(device)
    ended = time.perf_counter()
    if not torch.isfinite(reconstruction).all():
        raise FloatingPointError(f"Nonfinite reconstruction in {mode}")
    return reconstruction, estimates, {
        "sampling_and_calibration_seconds": sampled-start,
        "normalization_and_decode_seconds": ended-sampled,
        "receiver_seconds": ended-start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    required = ["encoder", "decoder", "infencoder", "icdm_s", "icdm_z"]
    missing = [manifest["weights"][key]["path"] for key in required if not Path(manifest["weights"][key]["path"]).is_file()]
    images, interference = files(manifest["images"]), files(manifest["interference_images"])
    if not images or not interference:
        missing.append("nonempty target and interference image directories")
    if missing:
        raise SystemExit("Preflight blocked: " + "; ".join(missing))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no experiment started")
    config = make_config(manifest)
    if args.preflight:
        print(json.dumps({"status": "input_paths_exist", "image_count": len(images), "interference_image_count": len(interference), "note": "Checkpoint compatibility is validated during strict model loading."}))
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata = {"manifest": manifest, "host": platform.node(), "torch": torch.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "sampler": {"steps":40, "order":3, "algorithm":"noise_prediction"}, "weights_sha256": {key:digest(manifest["weights"][key]["path"]) for key in required}, "status": "loading"}
    try:
        metadata["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        metadata["git_revision"] = None
    root = Path(__file__).resolve().parents[1]
    metadata["source_sha256"] = {str(p.relative_to(root)): digest(p) for directory in ["Autoencoder", "Diffusion", "DiT", "scripts"] for p in sorted((root/directory).rglob("*.py"))}
    meta_path = args.output / "metadata.json"
    records_path = args.output / "records.jsonl"
    completed = set()
    if args.resume and records_path.exists():
        for line_number, line in enumerate(records_path.read_text().splitlines(), 1):
            record = json.loads(line)
            key = (record["index"], record["seed"], record["snr"], record["true_global_sinr"], record["profile"], record["mode"])
            if key in completed:
                raise ValueError(f"Duplicate record at line {line_number}: {key}")
            completed.add(key)
    meta_path.write_text(json.dumps(metadata, indent=2))
    try:
        weights = manifest["weights"]
        architecture = config.MODEL.MODEL_NAME
        encoder = codec(architecture, "encoder", config, weights["encoder"]).to(device)
        decoder = codec(architecture, "decoder", config, weights["decoder"]).to(device)
        infencoder = codec(manifest["interference_model"], "encoder", config, weights["infencoder"]).to(device)
        from DiT.models import DiT_models
        kwargs = dict(in_channels=config.MODEL.OUT_CHANS, input_size=config.DATA.IMG_SIZE // 2**len(config.MODEL.DEPTHS))
        model_s = load_weights(DiT_models(**kwargs), weights["icdm_s"]).to(device)
        model_z = load_weights(DiT_models(**kwargs), weights["icdm_z"]).to(device)
        sampler = ICDMSampler(model_s, model_z, config).to(device)
        channel = Channel(config)
        limit = min(manifest.get("max_images", 32), len(images), len(interference))
        if limit < 1:
            raise ValueError("max_images must be positive")
        evaluation_batch_size = int(manifest.get("evaluation_batch_size", 1))
        if evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")
        modes = manifest.get("modes", ["A0", "A1", "A2", "A3"])
        unknown_modes = set(modes) - {"NO_ICDM", "A0", "A1", "A2", "A3", "A4"}
        if unknown_modes:
            raise ValueError(f"Unknown evaluation modes: {sorted(unknown_modes)}")
        metadata["actual_images_per_setting"] = limit
        metadata["evaluation_batch_size"] = evaluation_batch_size
        metadata["status"] = "running"
        meta_path.write_text(json.dumps(metadata, indent=2))
        warmed = set()
        stream_mode = "a" if args.resume else "x"
        with torch.no_grad(), records_path.open(stream_mode) as stream:
            for batch_start in range(0, limit, evaluation_batch_size):
                batch_end = min(batch_start + evaluation_batch_size, limit)
                batch_indices = list(range(batch_start, batch_end))
                image = torch.cat([load_image(images[index], manifest["image_size"], device) for index in batch_indices])
                inf_image = torch.cat([load_image(interference[index], manifest["interference_image_size"], device) for index in batch_indices])
                inf_image = torch.nn.functional.interpolate(inf_image, size=image.shape[-2:], mode="nearest")
                clean, inf = encoder(image), infencoder(inf_image)
                for trial_seed in manifest.get("seeds", [0, 1, 2]):
                    for snr, sinr in manifest["snr_sinr_pairs"]:
                        if sinr > snr:
                            raise ValueError("SINR must not exceed SNR")
                        for profile in manifest.get("profiles", ["stationary", "alternating", "burst"]):
                            block_size = manifest["block_size"]
                            one_power = block_powers(to_complex(clean)[0].numel(), block_size, snr, sinr, profile, device)
                            received_parts, h_parts = [], []
                            # Per-image channel calls preserve the original batch-one power
                            # normalization and make results invariant to evaluation batching.
                            for local_index, index in enumerate(batch_indices):
                                seed(trial_seed + 10000 * index)
                                if profile == "stationary":
                                    received_part, _, _, h_part = channel.inf_forward(clean[local_index:local_index+1], inf[local_index:local_index+1], snr, sinr)
                                else:
                                    received_part, _, _, h_part = channel.inf_forward_blockwise(clean[local_index:local_index+1], inf[local_index:local_index+1], snr, one_power, block_size)
                                received_parts.append(received_part)
                                h_parts.append(h_part)
                            received = torch.cat(received_parts)
                            h = torch.cat(h_parts)
                            power = one_power.expand(len(batch_indices), -1)
                            for mode in modes:
                                expected = {
                                    (index, trial_seed, snr, sinr, profile, mode)
                                    for index in batch_indices
                                }
                                if expected <= completed:
                                    continue
                                if mode not in warmed:
                                    seed(trial_seed + 10000 * batch_start + 1000000)
                                    recover(mode, sampler, decoder, received, h, snr, sinr, config.CHANNEL.TYPE, block_size, device, manifest.get("a4_min_alpha", .5))
                                    warmed.add(mode)
                                seed(trial_seed + 10000 * batch_start + 1000000)
                                reconstruction, estimates, timing = recover(mode, sampler, decoder, received, h, snr, sinr, config.CHANNEL.TYPE, block_size, device, manifest.get("a4_min_alpha", .5))
                                mse = (reconstruction.clamp(0, 1) - image).square().flatten(1).mean(1)
                                batch_count = len(batch_indices)
                                per_image_timing = {key: value / batch_count for key, value in timing.items()}
                                for local_index, index in enumerate(batch_indices):
                                    record_key = (index, trial_seed, snr, sinr, profile, mode)
                                    if record_key in completed:
                                        continue
                                    value = mse[local_index].item()
                                    record = {"image": str(images[index]), "interference_image": str(interference[index]), "index": index, "seed": trial_seed, "snr": snr, "true_global_sinr": sinr, "profile": profile, "mode": mode, "mse": value, "psnr": -10*math.log10(max(value, 1e-15)), **per_image_timing}
                                    record["receiver_batch_seconds"] = timing["receiver_seconds"]
                                    record["evaluation_batch_count"] = batch_count
                                    # Truth is recorded only by the experiment harness after reception.
                                    record["true_block_power"] = power[local_index:local_index+1].cpu().tolist()
                                    if estimates is not None:
                                        record["receiver_estimates"] = {key: tensor[local_index:local_index+1].cpu().tolist() for key, tensor in estimates.items()}
                                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                                    stream.flush()
                                    completed.add(record_key)
                                    print(json.dumps({key:record[key] for key in ["index", "seed", "snr", "true_global_sinr", "profile", "mode", "psnr", "receiver_seconds"]}), flush=True)
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
