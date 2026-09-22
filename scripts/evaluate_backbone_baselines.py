"""Independent confirmation of matched-retrained direct JSCC backbones."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baseline_metrics import ImageMetrics, METRIC_PROTOCOL
from scripts.run_phase1 import block_powers, digest, files, load_image, sync
from scripts.run_vtc2027 import keyseed
from scripts.train_backbone_baseline import build_models, from_complex, git_head, to_complex


def seeded(value: int) -> None:
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_architecture(name, external_root, checkpoint, device, image_size):
    args = SimpleNamespace(architecture=name, external_root=external_root, image_size=image_size)
    encoder, decoder, effective = build_models(args, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["architecture"] != name or state["role"] != "source":
        raise ValueError("Source checkpoint identity mismatch")
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval(); decoder.eval()
    return encoder, decoder, effective, state


def load_interference(name, external_root, checkpoint, device, image_size):
    args = SimpleNamespace(architecture=name, external_root=external_root, image_size=image_size)
    encoder, _, effective = build_models(args, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["architecture"] != name or state["role"] != "interference":
        raise ValueError("Interference checkpoint identity mismatch")
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    return encoder, effective, state


def encode(model, image, architecture, snr):
    latent = model(image) if architecture == "deepjscc" else model(image, snr)
    if architecture == "deepjscc" and latent.ndim == 3:
        latent = latent.unsqueeze(0)
    code = to_complex(latent, architecture)
    return code / code.abs().square().flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(1e-12).sqrt()


def decode(model, received, architecture, snr):
    normalized = received / received.abs().square().flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(1e-12).sqrt()
    real = from_complex(normalized, architecture)
    return model(real) if architecture == "deepjscc" else model(real, snr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deepjscc-root", type=Path, required=True)
    parser.add_argument("--mambajscc-root", type=Path, required=True)
    parser.add_argument("--deepjscc-source", type=Path, required=True)
    parser.add_argument("--deepjscc-interference", type=Path, required=True)
    parser.add_argument("--mambajscc-source", type=Path, required=True)
    parser.add_argument("--mambajscc-interference", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["stage"] != "confirmation" or manifest["evaluation_batch_size"] != 1:
        raise ValueError("Use the locked batch-one confirmation manifest")
    if manifest["channel_type"] != "awgn":
        raise ValueError("This confirmation is AWGN-only")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    offset, count = manifest["offset"], manifest["max_images"]
    image_paths = files(manifest["images"])[offset:offset + count]
    interference_paths = files(manifest["interference_images"])[offset:offset + count]
    if len(image_paths) != count or len(interference_paths) != count:
        raise ValueError("Incomplete confirmation inputs")
    architectures = {
        "deepjscc": dict(root=args.deepjscc_root, source=args.deepjscc_source,
                         interference=args.deepjscc_interference),
        "mambajscc": dict(root=args.mambajscc_root, source=args.mambajscc_source,
                          interference=args.mambajscc_interference),
    }
    scientific = {
        "manifest": manifest,
        "protocol_sha256": digest(Path("research/vtc2027_sheng/BACKBONE_BASELINE_PROTOCOL.md")),
        "evaluator_sha256": digest(Path(__file__)),
        "trainer_adapter_sha256": digest(Path("scripts/train_backbone_baseline.py")),
        "external_commits": {k: git_head(v["root"]) for k, v in architectures.items()},
        "weights_sha256": {f"{k}_{role}": digest(v[role]) for k, v in architectures.items()
                            for role in ["source", "interference"]},
        "images": [{"index": offset + i, "image": str(a), "image_sha256": digest(a),
                    "interference": str(b), "interference_sha256": digest(b)}
                   for i, (a, b) in enumerate(zip(image_paths, interference_paths))],
    }
    signature = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    expected = count * len(manifest["seeds"]) * len(manifest["conditions"]) * len(architectures)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    records = args.output / "records.jsonl"
    metadata = args.output / "metadata.json"
    done = set()
    if args.resume and metadata.exists():
        old = json.loads(metadata.read_text())
        if old["signature"] != signature:
            raise ValueError("Scientific inputs changed on resume")
        if records.exists():
            for line in records.open():
                row = json.loads(line)
                key = (row["architecture"], row["index"], row["seed"], row["condition"])
                if key in done:
                    raise ValueError("Duplicate record")
                done.add(key)
    meta = dict(scientific, signature=signature, status="loading", expected_records=expected,
                host=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                metrics=METRIC_PROTOCOL, started_at=time.time())
    atomic(metadata, meta)
    metric = ImageMetrics(device)
    lpips_path = Path(__import__("lpips").__file__).parent / "weights/v0.1/vgg.pth"
    vgg_path = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    meta["metric_weights_sha256"] = {str(p): digest(p) for p in [lpips_path, vgg_path]}
    try:
        mode = "a" if args.resume else "x"
        with torch.no_grad(), records.open(mode) as stream:
            for architecture, paths in architectures.items():
                source_encoder, decoder, effective, _ = load_architecture(
                    architecture, paths["root"], paths["source"], device, manifest["image_size"])
                interference_encoder, interference_effective, _ = load_interference(
                    architecture, paths["root"], paths["interference"], device, manifest["image_size"])
                meta.setdefault("effective_models", {})[architecture] = {
                    "source": effective, "interference": interference_effective,
                    "encoder_parameters": sum(p.numel() for p in source_encoder.parameters()),
                    "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
                }
                atomic(metadata, meta)
                for local, (image_path, interference_path) in enumerate(zip(image_paths, interference_paths)):
                    index = offset + local
                    image = load_image(image_path, manifest["image_size"], device)
                    interference_image = load_image(interference_path, manifest["interference_image_size"], device)
                    interference_image = torch.nn.functional.interpolate(
                        interference_image, size=image.shape[-2:], mode="nearest")
                    signal = encode(source_encoder, image, architecture, 20)
                    interference = encode(interference_encoder, interference_image, architecture, 20)
                    if signal.shape != interference.shape or signal[0].numel() != 1024:
                        raise ValueError(f"Latent/CBR mismatch for {architecture}: {signal.shape}")
                    for seed in manifest["seeds"]:
                        for condition_index, condition in enumerate(manifest["conditions"]):
                            key = (architecture, index, seed, condition_index)
                            if key in done:
                                continue
                            snr, sinr, profile = condition["snr"], condition["sinr"], condition["profile"]
                            power = block_powers(signal[0].numel(), manifest["block_size"], snr,
                                                 sinr, profile, device)
                            amplitude = power.sqrt().repeat_interleave(manifest["block_size"], 1)
                            amplitude = amplitude[:, :signal[0].numel()].reshape_as(signal)
                            seeded(keyseed("channel", seed, index, condition_index))
                            sigma = (1 / (2 * 10 ** (snr / 10))) ** 0.5
                            noise = sigma * (torch.randn_like(signal.real) + 1j * torch.randn_like(signal.real))
                            received = signal + interference * amplitude + noise
                            sync(device); began = time.perf_counter()
                            reconstruction = decode(decoder, received, architecture, snr)
                            sync(device); seconds = time.perf_counter() - began
                            if not torch.isfinite(reconstruction).all():
                                raise FloatingPointError(architecture)
                            values = metric(reconstruction, image)
                            row = dict(architecture=architecture, mode=f"{architecture.upper()}_DIRECT",
                                       index=index, seed=seed, condition=condition_index,
                                       snr=snr, sinr=sinr, profile=profile,
                                       receiver_seconds=seconds, nfe=0,
                                       received_sha256=tensor_hash(received),
                                       **{k: float(v) for k, v in values.items()})
                            stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                            done.add(key)
                            if local in [0, 15] and seed == manifest["seeds"][0] and condition_index in [0, 8, 15]:
                                folder = args.output / "qualitative" / architecture / f"{index}_{condition_index}"
                                folder.mkdir(parents=True, exist_ok=True)
                                for label, tensor in [("source", image), ("direct", reconstruction)]:
                                    array = (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
                                    Image.fromarray(array).save(folder / f"{label}.png")
                            if len(done) % 256 == 0:
                                print(json.dumps({"records": len(done), "expected": expected,
                                                  "architecture": architecture, "index": index}), flush=True)
                del source_encoder, decoder, interference_encoder
                torch.cuda.empty_cache()
        if len(done) != expected:
            raise ValueError(f"Incomplete record count: {len(done)} != {expected}")
        meta.update(status="complete", records=len(done), completed_at=time.time())
    except Exception as error:
        meta.update(status="failed", error=repr(error), failed_at=time.time())
        raise
    finally:
        atomic(metadata, meta)


if __name__ == "__main__":
    main()
