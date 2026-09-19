"""Paired validation of paper-derived receivers with four image metrics.

Only new outputs are accepted; existing records are never overwritten or mixed.
CDDM_RX is explicitly a shared-prior receiver port, not a full-system replication.
"""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from scripts.run_phase1 import (block_powers, codec, digest, files, load_image,
                               load_weights, make_config, oracle_parameters, seed, sync)
from scripts.baseline_metrics import ImageMetrics, METRIC_PROTOCOL
from scripts.cddm_receiver import cddm_receive
from Autoencoder.channel import Channel
from Diffusion import ICDMSampler
from Diffusion.calibration import equalize, to_complex
from DiT import create_diffusion
from DiT.models import DiT_models


def normalize(feature):
    return feature / (2 * feature.square().flatten(1).mean(1)).clamp_min(1e-12).sqrt().view(-1, 1, 1, 1)


@torch.no_grad()
def receive(mode, sampler, decoder, y, h, snr, sinr, block_size, abar, cddm_steps):
    info = {}
    if mode == "SWINJSCC":
        feature = equalize(y, h, snr, "awgn")
    elif mode == "ICDM_ORACLE":
        lam, beta = oracle_parameters(sinr, "awgn", y.device)
        feature, _ = sampler.SIC_sampling(snr, sinr, equalize(y, h, snr, "awgn"), h,
                                         Lambda=lam, Beta=beta, amplitude_mode="original")
    elif mode == "SC_ICDM_A4":
        feature, _, _ = sampler.SIC_sampling_alternating(snr, y, h, "awgn", block_size, min_alpha=.5)
    elif mode in {"CDDM_RX_N0", "CDDM_RX_ORACLE"}:
        # Oracle version is a separate information-advantaged control, not a blind method.
        power = 10 ** (-(snr if mode == "CDDM_RX_N0" else sinr) / 10)
        feature, info = cddm_receive(sampler.model_s, equalize(y, h, snr, "awgn"), power, abar, cddm_steps)
    else:
        raise ValueError(mode)
    feature = normalize(feature)
    rec = decoder(feature)
    if not torch.isfinite(rec).all():
        raise FloatingPointError(f"Nonfinite output in {mode}")
    return rec, feature, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/published_baselines_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.manifest.read_text())
    manifest = json.loads(Path(overrides.pop("base_manifest")).read_text())
    manifest.update(overrides)
    if args.smoke:
        manifest.update(max_images=4, seeds=[700], snr_sinr_pairs=[[20, -7], [20, 0], [20, 10]])
    if manifest["channel_type"] != "awgn":
        raise ValueError("CDDM port currently audited for AWGN only")
    images = files(manifest["images"])[:manifest["max_images"]]
    interferences = files(manifest["interference_images"])[:manifest["max_images"]]
    if len(images) != manifest["max_images"] or len(interferences) != len(images):
        raise ValueError("Requested complete paired image set is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    config = make_config(manifest)
    weights = manifest["weights"]
    root = Path(__file__).resolve().parents[1]
    meta = dict(status="loading", manifest=manifest, metrics=METRIC_PROTOCOL,
                study="engineering_smoke" if args.smoke else "exploratory_validation",
                gpu=torch.cuda.get_device_name(0), host=platform.node(), torch=torch.__version__,
                packages={p: importlib.metadata.version(p) for p in ["lpips", "torchvision", "pytorch-msssim"]},
                weights_sha256={k: digest(v["path"]) for k, v in weights.items()},
                image_pairs=[dict(index=i, image=str(a), image_sha256=digest(a), interference=str(b), interference_sha256=digest(b)) for i, (a, b) in enumerate(zip(images, interferences))],
                source_sha256={str(p.relative_to(root)): digest(p) for folder in ["Autoencoder", "DiT", "Diffusion", "scripts"] for p in sorted((root/folder).rglob("*.py"))})
    def save_meta():
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    save_meta()
    try:
        encoder = codec(config.MODEL.MODEL_NAME, "encoder", config, weights["encoder"]).to(device)
        decoder = codec(config.MODEL.MODEL_NAME, "decoder", config, weights["decoder"]).to(device)
        infencoder = codec(manifest["interference_model"], "encoder", config, weights["infencoder"]).to(device)
        kwargs = dict(in_channels=config.MODEL.OUT_CHANS, input_size=config.DATA.IMG_SIZE // 2 ** len(config.MODEL.DEPTHS))
        model_s = load_weights(DiT_models(**kwargs), weights["icdm_s"]).to(device)
        model_z = load_weights(DiT_models(**kwargs), weights["icdm_z"]).to(device)
        sampler = ICDMSampler(model_s, model_z, config).to(device)
        metrics = ImageMetrics(device)
        lpips_path = Path(__import__("lpips").__file__).parent / "weights/v0.1/vgg.pth"
        vgg_path = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
        meta["metric_weights_sha256"] = {str(p): digest(p) for p in [lpips_path, vgg_path]}
        # Identity check includes the actual pretrained perceptual network.
        probe = torch.rand(2, 3, 128, 128, device=device)
        identity = metrics(probe, probe)
        if identity["lpips_vgg"].abs().max() > 1e-6 or (identity["ms_ssim"] - 1).abs().max() > 1e-5:
            raise AssertionError("Metric identity gate failed")
        abar = create_diffusion("").alphas_cumprod
        channel = Channel(config)
        meta.update(status="running", metric_identity_gate="passed")
        save_meta()
        count = 0
        batch_size = manifest["evaluation_batch_size"]
        warmed = set()
        start_all = time.perf_counter()
        with torch.no_grad(), (args.output / "records.jsonl").open("x") as stream:
            for first in range(0, len(images), batch_size):
                ids = list(range(first, min(first + batch_size, len(images))))
                image = torch.cat([load_image(images[i], manifest["image_size"], device) for i in ids])
                inf_image = torch.cat([load_image(interferences[i], manifest["interference_image_size"], device) for i in ids])
                inf_image = torch.nn.functional.interpolate(inf_image, size=image.shape[-2:], mode="nearest")
                clean, inf = encoder(image), infencoder(inf_image)
                normalized_clean = normalize(clean)
                if first == 0:
                    meta["complex_channel_uses"] = clean[0].numel() // 2
                    meta["cbr"] = meta["complex_channel_uses"] / image[0].numel()
                    save_meta()
                for trial_seed in manifest["seeds"]:
                    for snr, sinr in manifest["snr_sinr_pairs"]:
                        for profile in manifest["profiles"]:
                            one_power = block_powers(to_complex(clean)[0].numel(), manifest["block_size"], snr, sinr, profile, device)
                            ys, hs = [], []
                            for local, i in enumerate(ids):
                                seed(trial_seed + 10000 * i)
                                if profile == "stationary":
                                    y, _, _, h = channel.inf_forward(clean[local:local+1], inf[local:local+1], snr, sinr)
                                else:
                                    y, _, _, h = channel.inf_forward_blockwise(clean[local:local+1], inf[local:local+1], snr, one_power, manifest["block_size"])
                                ys.append(y); hs.append(h)
                            y, h = torch.cat(ys), torch.cat(hs)
                            for mode in manifest["modes"]:
                                if mode not in warmed:
                                    seed(trial_seed + 10000 * first + 1000000)
                                    receive(mode, sampler, decoder, y, h, snr, sinr, manifest["block_size"], abar, manifest["cddm_steps"])
                                    warmed.add(mode)
                                seed(trial_seed + 10000 * first + 1000000)
                                sync(device); begin = time.perf_counter()
                                reconstruction, feature, info = receive(mode, sampler, decoder, y, h, snr, sinr, manifest["block_size"], abar, manifest["cddm_steps"])
                                sync(device); seconds = time.perf_counter() - begin
                                values = metrics(reconstruction, image)
                                values["latent_mse_real"] = (feature - normalized_clean).square().flatten(1).mean(1)
                                for local, i in enumerate(ids):
                                    record = dict(index=i, seed=trial_seed, snr=snr, sinr=sinr, profile=profile, mode=mode,
                                                  receiver_seconds=seconds / len(ids), receiver_batch_seconds=seconds, batch_count=len(ids), **info,
                                                  **{k: float(v[local]) for k, v in values.items()})
                                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                                    count += 1
                                stream.flush()
                                # Preselected qualitative pairs, not best-looking examples.
                                if first == 0 and trial_seed == manifest["seeds"][0] and sinr in [-7, 0, 10]:
                                    folder = args.output / "qualitative" / f"{profile}_{sinr}_{mode}"
                                    folder.mkdir(parents=True, exist_ok=True)
                                    for local in range(min(2, len(ids))):
                                        for label, tensor in [("source", image[local]), ("reconstruction", reconstruction[local])]:
                                            array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                                            Image.fromarray(array).save(folder / f"{ids[local]:04d}_{label}.png")
                print(json.dumps(dict(completed_images=ids[-1]+1, records=count, elapsed_seconds=time.perf_counter()-start_all)), flush=True)
        expected = len(images) * len(manifest["seeds"]) * len(manifest["snr_sinr_pairs"]) * len(manifest["profiles"]) * len(manifest["modes"])
        if count != expected:
            raise AssertionError((count, expected))
        meta.update(status="complete", records=count, elapsed_seconds=time.perf_counter()-start_all)
    except Exception as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        save_meta()


if __name__ == "__main__":
    main()
