from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"


def box(ax, x, y, w, h, text, fc="#f7f7f7", ec="#222222", fs=8, lw=1.0, weight="normal"):
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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#2b2b2b", lw=1.25, style="-|>", rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, title, fc="#fbfbfb", ec="#d5d9de"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.22, title, ha="left", va="center", fontsize=8.5, weight="bold")
    return patch


def small_bar(ax, x, y, w, n, color):
    gap = 0.025
    bw = (w - gap * (n - 1)) / n
    for i in range(n):
        box(ax, x + i * (bw + gap), y, bw, 0.08, "", fc=color, ec=color, lw=0.2)


def architecture() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    blue = "#2f6f9f"
    green = "#3f7d4a"
    orange = "#b66b2d"
    purple = "#7c5aa6"
    gray = "#4a4a4a"

    panel(ax, 0.25, 5.15, 15.45, 1.78, "A  ROI teacher backbone", fc="#f8fbfd")
    box(ax, 0.58, 5.82, 1.25, 0.48, "Expanded\nROI crop", fc="#ffffff", ec="#6f7f8f", fs=7.2, lw=0.9, weight="bold")
    ax.text(1.20, 5.62, "3 x 256 x 256", ha="center", va="center", fontsize=6.6, color=gray)
    arrow(ax, 1.86, 6.06, 2.25, 6.06, color=gray, lw=1.1)
    box(ax, 2.28, 5.73, 1.05, 0.64, "Patch\nembed", fc="#fff5e7", ec="#c88b4a", fs=7.0, lw=0.9)
    arrow(ax, 3.35, 6.06, 3.72, 6.06, color=gray, lw=1.1)

    stages = [
        ("Stage 1", "C=48", "stride 4", "4 blocks", "#e8f1fb"),
        ("Stage 2", "C=96", "stride 8", "4 blocks", "#e8f6ef"),
        ("Stage 3", "C=224", "stride 16", "15 blocks", "#fff3e8"),
        ("Stage 4", "C=448", "stride 32", "4 blocks", "#f1ecfb"),
    ]
    x = 3.78
    for i, (title, channels, stride, depth, fc) in enumerate(stages):
        box(ax, x, 5.58, 1.48, 0.95, "", fc=fc, ec="#66717d", lw=0.9)
        ax.text(x + 0.74, 6.32, title, ha="center", va="center", fontsize=7.8, weight="bold")
        ax.text(x + 0.74, 6.08, channels + ", " + stride, ha="center", va="center", fontsize=6.8, color="#333333")
        ax.text(x + 0.74, 5.84, depth, ha="center", va="center", fontsize=6.7, color="#4d4d4d")
        small_bar(ax, x + 0.22, 5.66, 1.04, min(6, int(depth.split()[0])), "#5d748c")
        if i < len(stages) - 1:
            arrow(ax, x + 1.50, 6.06, x + 1.78, 6.06, color=gray, lw=1.1)
        x += 1.78

    arrow(ax, 10.88, 6.06, 11.24, 6.06, color=gray, lw=1.1)
    box(ax, 11.28, 5.76, 0.76, 0.60, "GAP", fc="#e8f2ff", ec=blue, fs=7.1, lw=0.9)
    arrow(ax, 12.07, 6.06, 12.37, 6.06, color=gray, lw=1.1)
    box(ax, 12.42, 5.76, 1.28, 0.60, "1000-D task\nprojection", fc="#e8f2ff", ec=blue, fs=6.7, lw=0.9)
    arrow(ax, 13.73, 6.06, 14.00, 6.06, color=gray, lw=1.1)
    box(ax, 14.05, 5.76, 1.28, 0.60, "Binary head\nteacher logits", fc="#e8f2ff", ec=blue, fs=6.4, lw=0.9, weight="bold")

    panel(ax, 0.25, 2.53, 15.45, 2.32, "B  Representative block with modular adaptations", fc="#fdfcf9")
    box(ax, 0.58, 3.50, 0.72, 0.40, "x", fc="#ffffff", ec="#777777", fs=7.8)
    box(ax, 1.58, 3.43, 0.88, 0.54, "DPE", fc="#fff0de", ec=orange, fs=7.3, lw=0.9)
    box(ax, 2.78, 3.43, 0.92, 0.54, "Norm", fc="#f3f4f5", ec="#8c8c8c", fs=7.0, lw=0.8)
    box(ax, 4.02, 3.24, 1.72, 0.92, "Local-global\nmixer", fc="#efe8fb", ec=purple, fs=7.2, lw=0.9, weight="bold")
    box(ax, 4.28, 3.31, 1.20, 0.22, "DA option", fc="#fff1e4", ec=orange, fs=5.9, lw=0.7)
    box(ax, 6.20, 3.50, 0.44, 0.40, "+", fc="#ffffff", ec="#777777", fs=8.5)
    box(ax, 7.05, 3.43, 0.92, 0.54, "Norm", fc="#f3f4f5", ec="#8c8c8c", fs=7.0, lw=0.8)
    box(ax, 8.30, 3.24, 1.54, 0.92, "MS-FFN", fc="#e8f2ff", ec=blue, fs=7.5, lw=0.9, weight="bold")
    box(ax, 10.30, 3.50, 0.44, 0.40, "+", fc="#ffffff", ec="#777777", fs=8.5)
    box(ax, 11.28, 3.50, 0.80, 0.40, "y", fc="#ffffff", ec="#777777", fs=7.8)
    for x1, x2 in [(1.30, 1.56), (2.46, 2.76), (3.70, 4.00), (5.76, 6.18), (6.66, 7.02), (7.98, 8.28), (9.86, 10.28), (10.76, 11.26)]:
        arrow(ax, x1, 3.70, x2, 3.70, color=gray, lw=1.0)
    arrow(ax, 1.34, 3.93, 6.17, 3.93, color="#666666", lw=0.85, style="-", rad=0.0)
    arrow(ax, 1.34, 3.93, 1.34, 3.72, color="#666666", lw=0.85, style="-", rad=0.0)
    arrow(ax, 6.17, 3.93, 6.17, 3.73, color="#666666", lw=0.85, style="-|>", rad=0.0)
    arrow(ax, 6.72, 3.93, 10.27, 3.93, color="#666666", lw=0.85, style="-", rad=0.0)
    arrow(ax, 6.72, 3.93, 6.72, 3.72, color="#666666", lw=0.85, style="-", rad=0.0)
    arrow(ax, 10.27, 3.93, 10.27, 3.73, color="#666666", lw=0.85, style="-|>", rad=0.0)

    box(ax, 2.05, 2.82, 2.18, 0.33, "MCA: coordinate refinement", fc="#e8f3ff", ec=blue, fs=6.7, lw=0.8)
    arrow(ax, 3.16, 3.16, 3.16, 3.42, color=blue, lw=0.9)
    box(ax, 4.58, 2.82, 2.55, 0.33, "MUDD: stage-local dense reuse", fc="#eaf7ea", ec=green, fs=6.7, lw=0.8)
    arrow(ax, 5.88, 2.99, 4.80, 3.23, color=green, lw=0.9, rad=-0.15)
    box(ax, 7.56, 2.82, 2.30, 0.33, "DA: differential attention", fc="#fff1e4", ec=orange, fs=6.7, lw=0.8)
    arrow(ax, 8.70, 3.00, 5.35, 3.35, color=orange, lw=0.9, rad=0.14)
    box(ax, 12.85, 3.22, 2.30, 0.92, "Validation-fixed\nvariant analysis", fc="#ffffff", ec="#6f7f8f", fs=7.1, lw=0.9)
    arrow(ax, 12.10, 3.70, 12.82, 3.70, color=gray, lw=1.0)
    ax.text(13.99, 2.95, "modules are enabled or disabled\nonly before test evaluation", ha="center", va="center", fontsize=6.3, color="#555555")

    panel(ax, 0.25, 0.28, 15.45, 1.88, "C  Evaluation and deployment pathway", fc="#f9fbf8")
    workflow = [
        (0.62, 0.92, 1.58, "Teacher\nvariants", "#ffffff", "#6f7f8f"),
        (2.60, 0.92, 2.00, "ROI robustness\nGT / noisy / detector", "#fff7ec", orange),
        (5.08, 0.92, 1.58, "Case-level\npredictions", "#ffffff", "#6f7f8f"),
        (7.14, 0.92, 1.42, "KD loss\nCE + KL", "#eef6ff", blue),
        (9.04, 0.92, 2.12, "EfficientFormer-L1\n+ ECA student", "#e9f7ee", green),
        (11.74, 0.92, 1.70, "ONNX / TFLite\nexport", "#ffffff", "#6f7f8f"),
        (13.92, 0.92, 1.36, "Android\nlatency", "#eef6ff", blue),
    ]
    for wx, wy, ww, text, fc, ec in workflow:
        box(ax, wx, wy, ww, 0.58, text, fc=fc, ec=ec, fs=6.8, lw=0.85, weight="bold" if "student" in text else "normal")
    for i in range(len(workflow) - 1):
        x1 = workflow[i][0] + workflow[i][2]
        x2 = workflow[i + 1][0]
        arrow(ax, x1 + 0.03, 1.21, x2 - 0.03, 1.21, color=gray, lw=1.0)
    ax.text(4.46, 0.62, "analysis claim", ha="center", va="center", fontsize=6.3, color=orange)
    ax.text(10.10, 0.62, "deployment claim", ha="center", va="center", fontsize=6.3, color=green)

    fig.savefig(OUT / "fig_architecture_crop.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_architecture_crop.png", bbox_inches="tight", dpi=300)
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
