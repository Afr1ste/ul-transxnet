from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
    r"C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_reports\paper_complexity_performance_tradeoff.csv"
)
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["mean_auc_pct"] = df["mean_auc"] * 100.0
    df["bubble"] = 55 + df["params_m"] * 16

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.25,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    ax.set_facecolor("#fbfbf8")

    base = df[df["method"] != "UL-TransXNet"].copy()
    ours = df[df["method"] == "UL-TransXNet"].iloc[0]

    ax.scatter(
        base["macs_g"],
        base["mean_auc_pct"],
        s=base["bubble"],
        marker="o",
        facecolor="#b8c4cc",
        edgecolor="#3f4b55",
        linewidth=1.35,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        [ours["macs_g"]],
        [ours["mean_auc_pct"]],
        s=520,
        marker="*",
        facecolor="#c9292f",
        edgecolor="#7d0f17",
        linewidth=1.7,
        zorder=5,
    )

    label_offsets = {
        "MobileNetV3-Large": (0.16, 1.15, "left"),
        "EfficientNet-B0": (0.18, -1.25, "left"),
        "EfficientFormer-L1": (0.20, 1.15, "left"),
        "RepViT-M1.1": (0.18, -0.80, "left"),
        "DenseNet121": (0.20, -0.55, "left"),
        "ResNet50": (0.22, -0.80, "left"),
        "ConvNeXt-Tiny": (-1.45, 1.25, "left"),
        "Swin-T": (0.16, -1.20, "left"),
    }
    for _, row in base.iterrows():
        dx, dy, ha = label_offsets[row["method"]]
        ax.annotate(
            f"{row['method']}\n{row['params_m']:.1f}M",
            xy=(row["macs_g"], row["mean_auc_pct"]),
            xytext=(row["macs_g"] + dx, row["mean_auc_pct"] + dy),
            fontsize=9.2,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.9, color="#7a8791", shrinkA=2, shrinkB=4),
            zorder=4,
        )

    ax.annotate(
        "UL-TransXNet\n90.8% mean AUC\n14.4M / 2.36G",
        xy=(ours["macs_g"], ours["mean_auc_pct"]),
        xytext=(3.02, 91.35),
        fontsize=10.8,
        fontweight="bold",
        color="#7d0f17",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#7d0f17", shrinkA=3, shrinkB=5),
        zorder=6,
    )

    ax.text(
        0.03,
        0.05,
        "Bubble area denotes model parameters",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#4d5963",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d2d7dc", lw=0.8, alpha=0.9),
    )

    ax.set_xlabel("MACs (G)")
    ax.set_ylabel("Mean AUC (%)")
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(63.0, 93.0)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([65, 70, 75, 80, 85, 90])
    ax.minorticks_on()
    ax.grid(which="major", color="#b8b8b8", linewidth=0.85, alpha=0.75)
    ax.grid(which="minor", color="#dcdcdc", linewidth=0.55, linestyle="--", alpha=0.7)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.25)

    fig.tight_layout(pad=0.7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_teaser_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
