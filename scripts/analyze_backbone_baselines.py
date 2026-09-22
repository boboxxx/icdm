"""Audit and summarize independently retrained direct JSCC backbones."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vtc2027 import read_run

METRICS = ["psnr", "mse", "lpips_vgg", "ms_ssim", "receiver_seconds", "nfe"]
ARCHITECTURES = ["deepjscc", "mambajscc"]


def load_backbones(folder: Path):
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Backbone confirmation is not complete")
    rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    if len(rows) != meta["expected_records"] or len(rows) != meta["records"]:
        raise ValueError("Backbone record count mismatch")
    seen = set()
    for row in rows:
        key = (row["architecture"], row["index"], row["seed"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate backbone key: {key}")
        seen.add(key)
        if row["architecture"] not in ARCHITECTURES:
            raise ValueError(f"Unexpected architecture: {row['architecture']}")
        if not all(np.isfinite(row[name]) for name in METRICS):
            raise ValueError(f"Nonfinite metric at {key}")
    expected = {
        (architecture, item["index"], seed, condition)
        for architecture in ARCHITECTURES
        for item in meta["images"]
        for seed in meta["manifest"]["seeds"]
        for condition in range(len(meta["manifest"]["conditions"]))
    }
    if seen != expected:
        raise ValueError("Missing or unexpected backbone keys")
    return meta, rows


def summarize(rows):
    return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}


def clustered_contrast(target, reference, seed=20260922):
    deltas = defaultdict(list)
    for key in sorted(target):
        if key not in reference:
            raise ValueError(f"Reference key missing: {key}")
        deltas[key[0]].append(target[key]["psnr"] - reference[key]["psnr"])
    clusters = np.array([np.mean(value) for value in deltas.values()])
    rng = np.random.default_rng(seed)
    draws = clusters[rng.integers(0, len(clusters), size=(5000, len(clusters)))].mean(1)
    return {
        "mean_delta_psnr": float(clusters.mean()),
        "ci95_image_cluster": np.quantile(draws, [0.025, 0.975]).tolist(),
        "image_clusters": len(clusters),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    meta, rows = load_backbones(args.run)
    reference_meta, reference_rows, _ = read_run(args.reference)
    if reference_meta["manifest"]["stage"] != "confirmation":
        raise ValueError("Reference must be the independent confirmation run")
    reference = defaultdict(dict)
    for row in reference_rows:
        if row["mode"] in {"DIRECT", "SELECT"}:
            reference[row["mode"]][(row["index"], row["seed"], row["condition"])] = row
    report = {
        "status": "complete",
        "signature": meta["signature"],
        "reference_signature": reference_meta["signature"],
        "records": len(rows),
        "images": len(meta["images"]),
        "means": {},
        "by_sinr": {},
        "by_profile": {},
        "comparisons": {},
    }
    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture]
        report["means"][architecture] = summarize(selected)
        report["by_sinr"][architecture] = {
            str(sinr): summarize([row for row in selected if row["sinr"] == sinr])
            for sinr in sorted({row["sinr"] for row in selected})
        }
        report["by_profile"][architecture] = {
            profile: summarize([row for row in selected if row["profile"] == profile])
            for profile in sorted({row["profile"] for row in selected})
        }
        target = {(row["index"], row["seed"], row["condition"]): row for row in selected}
        report["comparisons"][architecture] = {
            mode.lower(): clustered_contrast(target, values)
            for mode, values in reference.items()
        }
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
