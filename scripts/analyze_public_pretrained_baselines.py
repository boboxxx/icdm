"""Audit public-pretrained diagnostic baselines and report descriptive contrasts."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_vtc2027 import read_run


MODELS = [
    "deepjscc_public_imagenet_snr19",
    "mambajscc_public_awgn10_clic2021",
    "mambajscc_public_rayleigh_div2k",
]
METRICS = ["psnr", "mse", "lpips_vgg", "ms_ssim", "receiver_seconds"]


def load_public(folder: Path):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("Public-pretrained run is not complete")
    rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    if len(rows) != meta.get("expected_records") or len(rows) != meta.get("records"):
        raise ValueError("Public-pretrained record count mismatch")
    seen = set()
    for row in rows:
        key = (row["model"], row["index"], row["seed"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate public-pretrained key: {key}")
        seen.add(key)
        if row["model"] not in MODELS:
            raise ValueError(f"Unexpected model: {row['model']}")
        if row.get("protocol") != "public pretrained; protocol mismatch":
            raise ValueError(f"Missing protocol-mismatch label at {key}")
        for name in METRICS + ["complex_channel_uses", "cbr"]:
            if not np.isfinite(row[name]):
                raise ValueError(f"Nonfinite {name} at {key}")
    expected = {
        (model, item["index"], seed, condition)
        for model in MODELS
        for item in meta["images"]
        for seed in meta["manifest"]["seeds"]
        for condition in range(len(meta["manifest"]["conditions"]))
    }
    if seen != expected:
        raise ValueError("Missing or unexpected public-pretrained keys")
    return meta, rows


def summarize(rows):
    result = {name: float(np.mean([row[name] for row in rows])) for name in METRICS}
    result["complex_channel_uses"] = float(np.mean([row["complex_channel_uses"] for row in rows]))
    result["cbr"] = float(np.mean([row["cbr"] for row in rows]))
    result["records"] = len(rows)
    return result


def descriptive_contrast(target, reference, seed=20260922):
    grouped = defaultdict(list)
    for key, row in target.items():
        if key not in reference:
            raise ValueError(f"Reference key missing: {key}")
        grouped[key[0]].append(row["psnr"] - reference[key]["psnr"])
    clusters = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = clusters[rng.integers(0, len(clusters), size=(5000, len(clusters)))].mean(axis=1)
    return {
        "mean_delta_psnr": float(clusters.mean()),
        "ci95_image_cluster": np.quantile(draws, [0.025, 0.975]).tolist(),
        "image_clusters": len(clusters),
        "interpretation": "descriptive only; public-pretrained protocol mismatch",
    }


def chinese_report(report):
    lines = [
        "# 公开预训练权重诊断结果",
        "",
        "本实验直接使用公开权重，不进行微调，并复用冻结确认集的 256 对图像、2 个种子和 18 种干扰条件。结果仅作为外部绝对参考，不与同协议重训练结果混合排名。",
        "",
        "| 模型 | 平均 PSNR (dB) | LPIPS↓ | MS-SSIM↑ | 复信道使用数 | CBR | 相对 SwinJSCC direct (dB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "deepjscc_public_imagenet_snr19": "DeepJSCC 公开 ImageNet/SNR19",
        "mambajscc_public_awgn10_clic2021": "MambaJSCC 公开 CLIC2021/AWGN10",
        "mambajscc_public_rayleigh_div2k": "MambaJSCC 公开 DIV2K/Rayleigh",
    }
    for model in MODELS:
        mean = report["means"][model]
        delta = report["comparisons"][model]["direct"]["mean_delta_psnr"]
        lines.append(
            f"| {labels[model]} | {mean['psnr']:.3f} | {mean['lpips_vgg']:.4f} | "
            f"{mean['ms_ssim']:.4f} | {mean['complex_channel_uses']:.0f} | "
            f"{mean['cbr']:.5f} | {delta:+.3f} |"
        )
    lines += [
        "",
        "解释边界：两组 MambaJSCC 权重保持 1/48 CBR，但训练数据、训练信道和训练分辨率与主协议不同；DeepJSCC 公开权重的 CBR 为 0.16256，约为目标 1/48 的 7.8 倍。上述差值是同一测试条件下的描述性参照，不是公平的算法优越性检验，也不支持声称复现或超过原始完整 CDDM/ICDM。",
        "",
        f"记录完整性：{report['records']} 条，{report['images']} 个图像组；签名 `{report['signature']}`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    meta, rows = load_public(args.run)
    reference_meta, reference_rows, _ = read_run(args.reference)
    if reference_meta["manifest"]["stage"] != "confirmation":
        raise ValueError("Reference must be the independent confirmation run")
    references = defaultdict(dict)
    for row in reference_rows:
        if row["mode"] in {"DIRECT", "SELECT"}:
            references[row["mode"]][(row["index"], row["seed"], row["condition"])] = row
    report = {
        "status": "complete",
        "scope": "public pretrained; protocol mismatch; descriptive external anchors only",
        "signature": meta["signature"],
        "reference_signature": reference_meta["signature"],
        "records": len(rows),
        "images": len(meta["images"]),
        "means": {},
        "by_sinr": {},
        "by_profile": {},
        "comparisons": {},
    }
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        report["means"][model] = summarize(selected)
        report["by_sinr"][model] = {
            str(sinr): summarize([row for row in selected if row["sinr"] == sinr])
            for sinr in sorted({row["sinr"] for row in selected})
        }
        report["by_profile"][model] = {
            profile: summarize([row for row in selected if row["profile"] == profile])
            for profile in sorted({row["profile"] for row in selected})
        }
        target = {(row["index"], row["seed"], row["condition"]): row for row in selected}
        report["comparisons"][model] = {
            name.lower(): descriptive_contrast(target, values)
            for name, values in references.items()
        }
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (args.run / "means.csv").open("w", newline="") as output:
        fields = ["model"] + METRICS + ["complex_channel_uses", "cbr", "records"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            writer.writerow({"model": model, **report["means"][model]})
    (args.run / "RESULTS_ZH.md").write_text(chinese_report(report))
    print(json.dumps({"status": "complete", "means": report["means"],
                      "comparisons": report["comparisons"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
