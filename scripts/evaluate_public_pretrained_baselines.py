"""Evaluate public DeepJSCC/MambaJSCC checkpoints without fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baseline_metrics import ImageMetrics, METRIC_PROTOCOL
from scripts.run_phase1 import block_powers, digest, files, load_image, sync
from scripts.run_vtc2027 import keyseed


def atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def seeded(value: int) -> None:
    random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_deep(root: Path, source: Path, checkpoint: Path, device):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("deepjscc_historical_model", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.DeepJSCC(c=19).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return model.encoder.eval().requires_grad_(False), model.decoder.eval().requires_grad_(False)


def legacy_timm_aliases() -> None:
    for name in ["drop", "mlp", "patch_embed", "weight_init", "helpers", "trace_utils",
                 "classifier", "adaptive_avgmax_pool", "norm", "create_act", "format"]:
        try:
            sys.modules[f"timm.models.layers.{name}"] = importlib.import_module(f"timm.layers.{name}")
        except Exception:
            pass


def load_mamba(root: Path, encoder_path: Path, decoder_path: Path, device):
    sys.path.insert(0, str(root / "selective_scan"))
    sys.path.insert(0, str(root / "adaptive_selective_scan"))
    sys.path.insert(0, str(root))
    legacy_timm_aliases()
    import models.vmamba as vmamba
    vmamba.selective_scan_cuda_core = importlib.import_module("selective_scan_cuda_core")
    encoder = torch.load(encoder_path, map_location=device, weights_only=False).eval().requires_grad_(False)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False).eval().requires_grad_(False)
    for model in [encoder, decoder]:
        for module in model.modules():
            if module.__class__.__name__ == "DropPath" and not hasattr(module, "scale_by_keep"):
                module.scale_by_keep = True
    return encoder, decoder


def encode(encoder, image, kind: str, model_snr: int):
    latent = encoder(image) if kind == "deep" else encoder(image, model_snr)
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    power = (2 * latent.square().mean()).clamp_min(1e-12)
    if kind == "deep":
        flat = latent.flatten(1)
        original = flat.shape[1]
        if original % 2:
            flat = torch.nn.functional.pad(flat, (0, 1))
        half = flat.shape[1] // 2
        code = torch.complex(flat[:, :half], flat[:, half:]) / power.sqrt()
        shape = tuple(latent.shape)
        return code, power, {"shape": shape, "original_real_values": original}
    half = latent.shape[2] // 2
    code = torch.complex(latent[:, :, :half], latent[:, :, half:]) / power.sqrt()
    return code, power, {"shape": tuple(latent.shape)}


def decode(decoder, received, power, shape, kind: str, model_snr: int):
    if kind == "deep":
        real = torch.cat((received.real, received.imag), 1)
        real = real[:, :shape["original_real_values"]].reshape(shape["shape"]) * power.sqrt()
        return decoder(real)
    real = torch.cat((received.real, received.imag), 2) * power.sqrt()
    return decoder(real, model_snr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-root", type=Path, required=True)
    parser.add_argument("--deep-source", type=Path, required=True)
    parser.add_argument("--deep-checkpoint", type=Path, required=True)
    parser.add_argument("--mamba-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["stage"] != "confirmation" or manifest["evaluation_batch_size"] != 1:
        raise ValueError("Locked batch-one confirmation manifest required")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.deterministic = True
    offset = manifest["offset"]
    count = 1 if args.smoke else manifest["max_images"]
    image_paths = files(manifest["images"])[offset:offset + count]
    interference_paths = files(manifest["interference_images"])[offset:offset + count]
    seeds = manifest["seeds"][:1] if args.smoke else manifest["seeds"]
    conditions = manifest["conditions"][:1] if args.smoke else manifest["conditions"]
    weight_paths = {
        "deep": args.deep_checkpoint,
        "mamba_awgn_encoder": args.pretrained_root / "encoder_awgn10_clic2021_out32.pt",
        "mamba_awgn_decoder": args.pretrained_root / "decoder_awgn10_clic2021_out32.pt",
        "mamba_rayleigh_encoder": args.pretrained_root / "encoder_rayleigh_multisnr_div2k_out32.pt",
        "mamba_rayleigh_decoder": args.pretrained_root / "decoder_rayleigh_multisnr_div2k_out32.pt",
    }
    extension_paths = list((args.mamba_root / "selective_scan").glob("selective_scan_cuda_core*.so"))
    extension_paths += list((args.mamba_root / "adaptive_selective_scan").glob("adaptive_selective_scan_cuda_core*.so"))
    scientific = {
        "manifest": manifest, "smoke": args.smoke,
        "protocol_sha256": digest(Path("research/vtc2027_sheng/PRETRAINED_BASELINE_PROTOCOL.md")),
        "evaluator_sha256": digest(Path(__file__)),
        "weights_sha256": {name: digest(path) for name, path in weight_paths.items()},
        "deep_historical_source_sha256": digest(args.deep_source),
        "cuda_extensions_sha256": {str(path): digest(path) for path in extension_paths},
        "images": [{"index": offset+i, "image": str(a), "image_sha256": digest(a),
                    "interference": str(b), "interference_sha256": digest(b)}
                   for i, (a, b) in enumerate(zip(image_paths, interference_paths))],
    }
    signature = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    models = {
        "deepjscc_public_imagenet_snr19": ("deep", 19),
        "mambajscc_public_awgn10_clic2021": ("mamba_awgn", 10),
        "mambajscc_public_rayleigh_div2k": ("mamba_rayleigh", 20),
    }
    expected = count * len(seeds) * len(conditions) * len(models)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    metadata_path = args.output / "metadata.json"; records_path = args.output / "records.jsonl"
    done = set()
    if args.resume and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["signature"] != signature: raise ValueError("Scientific signature changed")
        if records_path.exists():
            for line in records_path.open():
                row = json.loads(line); key = (row["model"], row["index"], row["seed"], row["condition"])
                if key in done: raise ValueError("Duplicate record")
                done.add(key)
    meta = dict(scientific, signature=signature, status="loading", expected_records=expected,
                host=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                metrics=METRIC_PROTOCOL, started_at=time.time())
    atomic(metadata_path, meta)
    metric = ImageMetrics(device)
    try:
        with torch.no_grad(), records_path.open("a" if args.resume else "x") as stream:
            for model_name, (variant, model_snr) in models.items():
                if variant == "deep":
                    encoder, decoder = load_deep(args.deep_root, args.deep_source, args.deep_checkpoint, device)
                    kind = "deep"
                else:
                    prefix = "awgn10_clic2021" if variant == "mamba_awgn" else "rayleigh_multisnr_div2k"
                    encoder, decoder = load_mamba(args.mamba_root,
                        args.pretrained_root / f"encoder_{prefix}_out32.pt",
                        args.pretrained_root / f"decoder_{prefix}_out32.pt", device)
                    kind = "mamba"
                for local, (image_path, interference_path) in enumerate(zip(image_paths, interference_paths)):
                    index = offset + local
                    image = load_image(image_path, 128, device)
                    interference_image = load_image(interference_path, 32, device)
                    interference_image = torch.nn.functional.interpolate(interference_image, (128, 128), mode="nearest")
                    signal, signal_power, shape = encode(encoder, image, kind, model_snr)
                    interference, _, _ = encode(encoder, interference_image, kind, model_snr)
                    uses = signal[0].numel(); cbr = uses / image[0].numel()
                    if signal.shape != interference.shape: raise ValueError("Latent shape mismatch")
                    for seed in seeds:
                        for condition_index, condition in enumerate(conditions):
                            key = (model_name, index, seed, condition_index)
                            if key in done: continue
                            snr, sinr, profile = condition["snr"], condition["sinr"], condition["profile"]
                            power = block_powers(uses, manifest["block_size"], snr, sinr, profile, device)
                            amplitude = power.sqrt().repeat_interleave(manifest["block_size"], 1)[:, :uses].reshape_as(signal)
                            seeded(keyseed("channel", seed, index, condition_index))
                            sigma = (1 / (2 * 10 ** (snr / 10))) ** 0.5
                            noise = sigma * (torch.randn_like(signal.real) + 1j * torch.randn_like(signal.real))
                            received = signal + amplitude * interference + noise
                            sync(device); began = time.perf_counter()
                            reconstruction = decode(decoder, received, signal_power, shape, kind, model_snr)
                            sync(device); seconds = time.perf_counter() - began
                            values = metric(reconstruction, image)
                            row = dict(model=model_name, protocol="public pretrained; protocol mismatch",
                                       index=index, seed=seed, condition=condition_index, snr=snr,
                                       sinr=sinr, profile=profile, image_size=128,
                                       complex_channel_uses=uses, cbr=cbr, receiver_seconds=seconds,
                                       received_sha256=tensor_hash(received),
                                       **{name: float(value) for name, value in values.items()})
                            stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush(); done.add(key)
                del encoder, decoder; torch.cuda.empty_cache()
        if len(done) != expected: raise ValueError(f"Incomplete: {len(done)} != {expected}")
        meta.update(status="complete", records=len(done), completed_at=time.time())
    except Exception as error:
        meta.update(status="failed", error=repr(error), failed_at=time.time()); raise
    finally:
        atomic(metadata_path, meta)


if __name__ == "__main__": main()
