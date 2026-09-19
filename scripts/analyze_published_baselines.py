"""Audit complete paired receiver records and write reproducible summaries."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

import numpy as np

LABELS = {
    "SWINJSCC": "SwinJSCC（固定编码器）",
    "CDDM_RX_N0": "CDDM 接收算法移植（仅 N0）",
    "CDDM_RX_ORACLE": "CDDM 接收算法移植（全局 SINR oracle）",
    "ICDM_ORACLE": "ICDM（全局 SINR oracle）",
    "SC_ICDM_A4": "SC-ICDM A4（未知 SINR）",
}
METRICS = ["psnr", "mse", "lpips_vgg", "ms_ssim", "ms_ssim_db", "latent_mse_real", "receiver_seconds"]
KEY = ["index", "seed", "snr", "sinr", "profile", "mode"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def write_raw_records_csv(directory, records):
    fields = list(records[0])
    fields.extend(sorted(set().union(*(record.keys() for record in records)) - set(fields)))
    with (directory / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_reconstruction_manifest(directory, manifest):
    root = directory / "qualitative"
    rows = []
    for path in sorted(root.glob("*/*.png")):
        condition = path.parent.name
        profile = next((p for p in manifest["profiles"] if condition.startswith(p + "_")), None)
        if profile is None:
            raise ValueError(f"Unknown qualitative condition: {condition}")
        sinr_text, mode = condition[len(profile) + 1:].split("_", 1)
        stem, kind = path.stem.rsplit("_", 1)
        width, height = png_size(path)
        rows.append({
            "relative_path": path.relative_to(directory).as_posix(),
            "profile": profile,
            "sinr_db": int(sinr_text),
            "mode": mode,
            "sample_index": int(stem),
            "kind": kind,
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    expected = len(manifest["profiles"]) * 3 * len(manifest["modes"]) * 2 * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} qualitative PNGs, found {len(rows)}")
    with (directory / "RECONSTRUCTION_MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_bundle_manifest(directory):
    output = directory / "BUNDLE_MANIFEST.csv"
    rows = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p != output):
        rows.append({
            "relative_path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--smoke-reference", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.directory / "metadata.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Refuse to summarize an incomplete run as a complete experiment")
    records = [json.loads(line) for line in (args.directory / "records.jsonl").read_text().splitlines()]
    m = meta["manifest"]
    index = {tuple(r[k] for k in KEY): r for r in records}
    expected = {(i, s, snr, sinr, p, mode) for i, s, (snr, sinr), p, mode in itertools.product(range(m["max_images"]), m["seeds"], m["snr_sinr_pairs"], m["profiles"], m["modes"])}
    assert len(index) == len(records) == len(expected) == meta["records"]
    assert set(index) == expected
    for r in records:
        assert all(math.isfinite(r[k]) for k in METRICS)
        assert 0 <= r["ms_ssim"] <= 1.000001 and r["lpips_vgg"] >= -1e-8
        assert abs(r["psnr"] + 10 * math.log10(max(r["mse"], 1e-12))) < 1e-4
    write_raw_records_csv(args.directory, records)
    write_reconstruction_manifest(args.directory, m)
    audit = dict(status="passed", records=len(records), image_clusters=m["max_images"], complete_unique_pairing=True)
    if args.smoke_reference:
        smoke = [json.loads(line) for line in (args.smoke_reference / "records.jsonl").read_text().splitlines()]
        smeta = json.loads((args.smoke_reference / "metadata.json").read_text())
        assert smeta["weights_sha256"] == meta["weights_sha256"]
        assert smeta["image_pairs"] == meta["image_pairs"][:4]
        maxerr = max(abs(r[k] - index[tuple(r[t] for t in KEY)][k]) for r in smoke for k in METRICS if k != "receiver_seconds")
        assert maxerr < 2e-5, maxerr
        audit["smoke_repeat_max_abs_error"] = maxerr

    def aggregate(rows):
        result = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
        result["psnr_from_mean_mse_not_mean_psnr"] = -10 * math.log10(result["mse"])
        result["lpips_db_of_mean"] = 10 * math.log10(max(result["lpips_vgg"], 1e-12))
        result["n"] = len(rows)
        return result

    summary = {mode: aggregate([r for r in records if r["mode"] == mode]) for mode in m["modes"]}
    strata = [dict(mode=mode, profile=profile, sinr=sinr,
                   **aggregate([r for r in records if r["mode"] == mode and r["profile"] == profile and r["sinr"] == sinr]))
              for mode, profile, (_, sinr) in itertools.product(m["modes"], m["profiles"], m["snr_sinr_pairs"])]
    paired = {}
    if meta["study"] != "engineering_smoke":
        for baseline in m["modes"]:
            if baseline == "SC_ICDM_A4":
                continue
            paired[baseline] = {}
            for metric in ["psnr", "mse", "lpips_vgg", "ms_ssim"]:
                diffs = []
                wins = []
                for i, s, (snr, sinr), p in itertools.product(range(m["max_images"]), m["seeds"], m["snr_sinr_pairs"], m["profiles"]):
                    prefix = (i, s, snr, sinr, p)
                    difference = index[prefix + ("SC_ICDM_A4",)][metric] - index[prefix + (baseline,)][metric]
                    diffs.append(difference)
                    wins.append(difference > 0 if metric in {"psnr", "ms_ssim"} else difference < 0)
                paired[baseline][metric] = dict(
                    ours_minus_baseline=float(np.mean(diffs)),
                    favorable_pair_fraction=float(np.mean(wins)),
                    matched_pairs=len(diffs),
                )
    by_sinr = [dict(mode=mode, sinr=sinr,
                    **aggregate([r for r in records if r["mode"] == mode and r["sinr"] == sinr]))
               for mode, (_, sinr) in itertools.product(m["modes"], m["snr_sinr_pairs"])]
    by_profile = [dict(mode=mode, profile=profile,
                       **aggregate([r for r in records if r["mode"] == mode and r["profile"] == profile]))
                  for mode, profile in itertools.product(m["modes"], m["profiles"])]
    result = dict(audit=audit, metadata_study=meta["study"], means=summary, strata=strata,
                  by_sinr=by_sinr, by_profile=by_profile,
                  paired_descriptive_effects=paired,
                  uncertainty_reporting="none: user requested no confidence intervals and no standard-deviation summaries")
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2))
    for name, rows in [("means", [dict(mode=k, **v) for k, v in summary.items()]),
                       ("strata", strata), ("by_sinr", by_sinr), ("by_profile", by_profile)]:
        with (args.directory / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    lines = ["# 已发表方法对照：配对验证结果", "", f"状态：{meta['study']}；{m['max_images']} 对图像，{len(m['seeds'])} 个固定信道种子，{len(records):,} 条完整记录。",
             f"GPU：{meta['gpu']}；CBR={meta['cbr']:.8f}（{meta['complex_channel_uses']} 个复信道符号/图像）。", "",
             "| 接收方法 | PSNR ↑ (dB) | 图像 MSE ↓ | LPIPS-VGG ↓ | MS-SSIM ↑ | 接收时间 (ms/图像) |", "|---|---:|---:|---:|---:|---:|"]
    for mode, v in summary.items():
        lines.append(f"| {LABELS[mode]} | {v['psnr']:.3f} | {v['mse']:.6f} | {v['lpips_vgg']:.4f} | {v['ms_ssim']:.4f} | {1000*v['receiver_seconds']:.1f} |")
    lines += ["", "CDDM 为同一 DiT 先验、固定解码器上的 40 次调用接收算法移植，不是原论文三阶段系统的完整复现。oracle 表示接收机获得真实全局 SINR；不是盲接收机。",
              "", "PSNR 先逐图计算再平均；MSE 是 [0,1] RGB 图像误差；LPIPS 使用 VGG v0.1 和 [-1,1] 输入；MS-SSIM 是明确声明的四尺度标准乘积，不沿用旧代码的重复末尺度乘法。",
              "", "时间在同一轮交错运行、预热后测量，包括采样/归一化/解码，不包括编码和指标计算；batch=4。运行期间检测到其他项目在同一主机训练，因此时间只作诊断记录，不能作为独占硬件效率比较，也不能与论文的 batch=1 延迟直接比较。"]
    if paired:
        lines += ["", "## 配对描述性差值", "", "差值均为 A4 − 对照；PSNR/MS-SSIM 为正有利，MSE/LPIPS 为负有利。按用户要求不计算置信区间，也不报告标准差。括号内为 2,304 个严格配对条件中 A4 指标更优的比例。", "", "| 对照 | PSNR 差值（有利比例） | LPIPS 差值（有利比例） | MS-SSIM 差值（有利比例） |", "|---|---|---|---|"]
        for mode, effects in paired.items():
            cells = []
            for metric in ["psnr", "lpips_vgg", "ms_ssim"]:
                e = effects[metric]
                cells.append(f"{e['ours_minus_baseline']:+.4f} ({100*e['favorable_pair_fraction']:.1f}%)")
            lines.append("| " + LABELS[mode] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 解释边界", "", "这是验证集结果，不是独立测试。不能据此声称全面超越已发表完整系统。模型权重、数据对、代码和感知指标权重的哈希见 metadata.json；全部分 SINR/干扰形态结果见 strata.csv。",
              "", "本报告只使用均值、严格配对差值和更优比例，不包含置信区间或标准差。基线选择与原文差异见 ../PUBLISHED_BASELINE_PROTOCOL.md。"]
    (args.directory / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_bundle_manifest(args.directory)
    print(json.dumps(dict(audit=audit, means=summary), indent=2))


if __name__ == "__main__":
    main()
