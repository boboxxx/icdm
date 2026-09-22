"""Plot the existing image-cluster audit; no fitted or synthetic results."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
audit = json.loads((ROOT / "research/a4_image_cluster_audit.json").read_text())
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .6,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
})

fig, ax = plt.subplots(figsize=(3.5, 2.35))
styles = [("stationary", "Stationary", "#0072B2", "o", "-"),
          ("alternating", "Alternating", "#D55E00", "s", "--"),
          ("burst", "Burst", "#009E73", "^", "-.")]
for profile, label, color, marker, style in styles:
    rows = sorted([r for r in audit["contrasts"] if r["reference"] == "A3"
                   and r["profile"] == profile and r["sinr"] is not None], key=lambda r: r["sinr"])
    x = np.array([r["sinr"] for r in rows])
    y = np.array([r["mean"] for r in rows])
    lo = np.array([r["ci95"][0] for r in rows])
    hi = np.array([r["ci95"][1] for r in rows])
    ax.errorbar(x, y, yerr=np.array([y-lo, hi-y]), color=color, marker=marker,
                linestyle=style, capsize=2, elinewidth=.7, label=label)
ax.axhline(0, color="0.3", linewidth=.7, linestyle=":")
ax.set(xlabel="Nominal SINR (dB)", ylabel="PSNR gain over block-blind (dB)",
       xticks=[-7, -4, 0, 4, 7, 10], ylim=(-.32, 1.57))
ax.grid(axis="y", alpha=.17, linewidth=.5)
ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=1.5,
          columnspacing=.8, handletextpad=.4)
fig.tight_layout(pad=.5)
fig.savefig(OUT / "fig_calibration.pdf", bbox_inches="tight", pad_inches=.025)
fig.savefig(OUT / "fig_calibration.png", dpi=300, bbox_inches="tight", pad_inches=.025)
print(OUT / "fig_calibration.pdf")

totals = defaultdict(lambda: [0.0, 0])
for folder in ("sheng_phase1_full", "sheng_a4_full"):
    with (ROOT / "research" / folder / "records.jsonl").open() as records:
        for line in records:
            row = json.loads(line)
            key = (row["mode"], row["profile"], row["true_global_sinr"])
            totals[key][0] += row["psnr"]
            totals[key][1] += 1

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharex=True, sharey=True)
methods = [
    ("NO_ICDM", "Direct", "0.2", "x", ":"),
    ("A0", "Global-SINR", "#0072B2", "s", "--"),
    ("A2", "Global-blind", "#56B4E9", "^", "-."),
    ("A3", "Block-blind", "#009E73", "D", "--"),
    ("A4", "SC-ICDM", "#D55E00", "o", "-"),
]
sinrs = [-7, -4, 0, 4, 7, 10]
for ax, profile, tag in zip(axes, ("stationary", "alternating", "burst"), ("a", "b", "c")):
    for mode, label, color, marker, style in methods:
        groups = [totals[(mode, profile, snr)] for snr in sinrs]
        assert all(n == 600 for _, n in groups), (mode, profile, groups)
        means = [s/n for s, n in groups]
        ax.plot(sinrs, means, color=color, marker=marker, linestyle=style,
                linewidth=1.5 if mode == "A4" else 1.0,
                markerfacecolor=color if mode == "A4" else "white", label=label)
    ax.set_xlabel(f"({tag}) {profile.capitalize()}\nNominal SINR (dB)")
    ax.set_xticks(sinrs)
    ax.grid(axis="y", alpha=.17, linewidth=.5)
axes[0].set_ylabel("Reconstruction PSNR (dB)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(.51, 1.0), columnspacing=1.7)
fig.tight_layout(rect=(0, 0, 1, .89), pad=.6, w_pad=1.0)
fig.savefig(OUT / "fig_psnr.pdf", bbox_inches="tight", pad_inches=.025)
fig.savefig(OUT / "fig_psnr.png", dpi=300, bbox_inches="tight", pad_inches=.025)
print(OUT / "fig_psnr.pdf")
