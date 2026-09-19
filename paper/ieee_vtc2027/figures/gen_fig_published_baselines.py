"""Plot audited paired receiver results; no hard-coded measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "SWINJSCC": ("SwinJSCC", "#777777", "o", "--"),
    "CDDM_RX_N0": ("CDDM-RX (N0)", "#56B4E9", "v", ":"),
    "CDDM_RX_ORACLE": ("CDDM-RX (oracle)", "#E69F00", "^", "-."),
    "ICDM_ORACLE": ("ICDM (oracle)", "#009E73", "s", "--"),
    "SC_ICDM_A4": ("SC-ICDM A4", "#D55E00", "D", "-"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    assert summary["audit"]["status"] == "passed"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8.5, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    metrics = [("psnr", "PSNR (dB), higher is better", 1),
               ("mse", "Image MSE ($10^{-2}$), lower", 100),
               ("lpips_vgg", "LPIPS-VGG, lower is better", 1),
               ("ms_ssim", "MS-SSIM, higher is better", 1)]
    fig, axes = plt.subplots(3, 4, figsize=(10.7, 6.7), sharex=True)
    profiles = ["stationary", "alternating", "burst"]
    for row, profile in enumerate(profiles):
        for col, (metric, label, factor) in enumerate(metrics):
            ax = axes[row, col]
            for mode, (name, color, marker, style) in STYLES.items():
                rows = sorted([r for r in summary["strata"] if r["profile"] == profile and r["mode"] == mode], key=lambda r: r["sinr"])
                ax.plot([r["sinr"] for r in rows], [factor*r[metric] for r in rows], label=name,
                        color=color, marker=marker, linestyle=style, markersize=3.8,
                        linewidth=1.9 if mode == "SC_ICDM_A4" else 1.25)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(profile.capitalize())
            if row == 2:
                ax.set_xlabel("SINR (dB)")
            ax.set_xticks([-7, -4, 0, 4, 7, 10])
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.005))
    fig.tight_layout(rect=(0, .03, 1, .96), h_pad=1.2, w_pad=1.1)
    study = "Engineering smoke only" if summary["metadata_study"] == "engineering_smoke" else "Exploratory paired validation"
    fig.text(.5, .005, study + ". CDDM-RX: shared-prior, frozen-decoder port; oracle: true global SINR.", ha="center", fontsize=8)
    fig.savefig(args.directory / "fig_four_metrics.pdf", bbox_inches="tight")
    fig.savefig(args.directory / "fig_four_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # A separate, physically sized IEEE full-width figure remains readable
    # without shrinking the 12-panel diagnostic dashboard into two columns.
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.9))
    for ax, (metric, label, factor) in zip(axes.flat, metrics):
        for mode, (name, color, marker, style) in STYLES.items():
            rows = [r for r in summary["strata"] if r["mode"] == mode]
            sinrs = sorted({r["sinr"] for r in rows})
            means = [np.mean([r[metric] for r in rows if r["sinr"] == s]) * factor for s in sinrs]
            ax.plot(sinrs, means, label=name, color=color, marker=marker, linestyle=style,
                    markersize=3.5, linewidth=1.8 if mode == "SC_ICDM_A4" else 1.2)
        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel(label.split(",")[0])
        ax.set_xticks([-7, -4, 0, 4, 7, 10])
        ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .92), h_pad=1, w_pad=1.2)
    fig.text(.5, .005, study + "; equal averaging over three interference profiles.", ha="center", fontsize=8)
    fig.savefig(args.directory / "fig_metrics_overall.pdf", bbox_inches="tight")
    fig.savefig(args.directory / "fig_metrics_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
