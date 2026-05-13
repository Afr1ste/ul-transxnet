from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"


def box(ax, x, y, w, h, text, fc="#f7f7f7", ec="#222222", fs=8, lw=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#222222", shrinkA=2, shrinkB=2),
    )


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.55))
    ax.set_xlim(0, 14.75)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    ax.text(0.25, 4.10, "Backbone flow", fontsize=9, weight="bold", ha="left", va="center")
    box(ax, 0.25, 2.85, 1.2, 0.68, "ROI crop\n3 x 256 x 256", fc="#ffffff", fs=7.5)
    arrow(ax, 1.48, 3.19, 1.90, 3.19)

    stages = [
        ("Stage 1", "C=48, s=4", "4 blocks"),
        ("Stage 2", "C=96, s=8", "4 blocks"),
        ("Stage 3", "C=224, s=16", "15 blocks"),
        ("Stage 4", "C=448, s=32", "4 blocks"),
    ]
    x = 1.95
    for i, (title, dim, depth) in enumerate(stages):
        box(ax, x, 2.62, 1.75, 1.08, "", fc="#fbfbfb", ec="#333333", lw=1.0)
        ax.text(x + 0.875, 3.47, title, ha="center", va="center", fontsize=8.6, weight="bold")
        ax.text(x + 0.875, 3.17, dim, ha="center", va="center", fontsize=7.6)
        ax.text(x + 0.875, 2.89, depth, ha="center", va="center", fontsize=7.4, color="#444444")
        if i < len(stages) - 1:
            arrow(ax, x + 1.78, 3.19, x + 2.08, 3.19)
        x += 2.1

    arrow(ax, 10.38, 3.19, 10.75, 3.19)
    box(ax, 10.82, 2.88, 0.95, 0.62, "GAP", fc="#dfefff", fs=8)
    arrow(ax, 11.78, 3.19, 12.10, 3.19)
    box(ax, 12.12, 2.88, 1.23, 0.62, "1000-D\nprojection", fc="#e9f4ff", fs=7.4)
    arrow(ax, 13.36, 3.19, 13.63, 3.19)
    box(ax, 13.66, 2.88, 0.85, 0.62, "MLP\n2-way", fc="#e9f4ff", fs=7.4)

    ax.text(0.25, 1.86, "Representative block and reported adaptations", fontsize=9, weight="bold", ha="left", va="center")
    box(ax, 0.25, 0.38, 13.45, 1.18, "", fc="#fffdf8", ec="#333333", lw=1.0)
    block_steps = [
        (0.75, "Input", "#ffffff"),
        (1.65, "DPE", "#ffe8d6"),
        (2.52, "Norm", "#f6f6f6"),
        (3.38, "Mixer", "#ead7ff"),
        (4.82, "Residual", "#ffffff"),
        (5.82, "Norm", "#f6f6f6"),
        (6.70, "MS-FFN", "#d9edf7"),
        (7.87, "Output", "#ffffff"),
    ]
    for sx, label, color in block_steps:
        box(ax, sx, 0.86, 0.82, 0.36, label, fc=color, fs=7.1)
    for sx in [1.57, 2.44, 3.30, 4.20, 5.64, 6.62, 7.52]:
        arrow(ax, sx, 1.04, sx + 0.20, 1.04)

    box(ax, 8.95, 1.02, 3.65, 0.34, "MCA: coordinate-aware refinement", fc="#e8f3ff", fs=7.3, ec="#4077aa")
    box(ax, 8.95, 0.67, 3.65, 0.34, "MUDD: dynamic dense reuse", fc="#eaf7ea", fs=7.3, ec="#4f8f4f")
    box(ax, 8.95, 0.32, 3.65, 0.34, "DA: differential attention", fc="#fff1e5", fs=7.3, ec="#bb6b2c")
    ax.text(13.00, 0.84, "Enabled/disabled\nas model variants", fontsize=7.0, ha="left", va="center", color="#333333")
    fig.savefig(OUT / "fig_architecture_crop.pdf", bbox_inches="tight")
    plt.close(fig)


def roi_robustness() -> None:
    src = ROOT / "results" / "strict_20260514" / "roi_robustness" / "roi_robustness_metrics_public.csv"
    colors = {
        "TransXNet-MUDD+DA": "#0072B2",
        "TransXNet-base": "#D55E00",
        "ConvNeXt-Tiny": "#009E73",
        "Swin-T": "#CC79A7",
    }
    markers = {
        "TransXNet-MUDD+DA": "o",
        "TransXNet-base": "s",
        "ConvNeXt-Tiny": "^",
        "Swin-T": "D",
    }
    order = ["GT ROI", "GT ROI + 5% noise", "GT ROI + 10% noise", "GT ROI + 20% noise", "Detector ROI"]
    labels = ["GT", "5%", "10%", "20%", "Det."]
    rows: dict[str, dict[str, float]] = {}
    with src.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            rows.setdefault(row["model"], {})[row["condition"]] = float(row["auc"])

    x = list(range(len(order)))
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.2, 2.45),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34},
    )
    for name, values in rows.items():
        auc = [values[c] for c in order]
        ax1.plot(
            x,
            auc,
            marker=markers[name],
            lw=1.7,
            ms=3.8,
            color=colors[name],
            label=name,
        )
        drop = [v - auc[0] for v in auc]
        ax2.plot(
            x,
            drop,
            marker=markers[name],
            lw=1.7,
            ms=3.8,
            color=colors[name],
        )

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.tick_params(axis="x", labelrotation=0)
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.86, 0.975)
    ax1.set_title("(a) Absolute AUC", fontsize=9)
    ax2.axhline(0.0, color="#333333", lw=0.8, alpha=0.65)
    ax2.set_ylabel("Delta AUC vs. GT")
    ax2.set_ylim(-0.075, 0.008)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: "0.00" if abs(y) < 5e-4 else f"{y:.2f}"))
    ax2.set_title("(b) Robustness loss", fontsize=9)
    ax1.legend(frameon=False, loc="lower left", fontsize=7, ncol=1)
    fig.text(0.5, -0.035, "ROI source / perturbation", ha="center", va="center", fontsize=9)
    fig.savefig(OUT / "fig_roi_robustness_curve_crop.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    architecture()
    roi_robustness()


if __name__ == "__main__":
    main()
