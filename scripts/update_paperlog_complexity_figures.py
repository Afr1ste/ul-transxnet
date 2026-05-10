"""Regenerate complexity figures from recomputed paper-log-label metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    REPO_ROOT
    / "results"
    / "provenance_release_20260510"
    / "predictions"
    / "recomputed_paperlog_labels"
    / "full_complexity_pool_recomputed.csv"
)
FIG_DIR = REPO_ROOT / "paper" / "figures"


def bubble_area(params_m: float) -> float:
    return 28.0 + float(params_m) * 25.0


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("", "_crop"):
        fig.savefig(FIG_DIR / f"{stem}{suffix}.pdf", bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    df = pd.read_csv(METRICS)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    base = df[df["method"] != "UL-TransXNet"]
    ax.scatter(
        base["macs_g"],
        base["mean_auc"] * 100.0,
        s=[bubble_area(v) for v in base["params_m"]],
        facecolor="#b9c6cf",
        edgecolor="#3a4b57",
        linewidth=1.1,
        alpha=0.95,
        zorder=2,
    )
    for row in base.itertuples(index=False):
        dx = 0.06
        dy = 0.1
        if row.method in {"Swin-T", "ConvNeXt-Tiny"}:
            dy = -1.0
        ax.text(
            row.macs_g + dx,
            row.mean_auc * 100.0 + dy,
            f"{row.method}\n{row.params_m:.1f}M",
            fontsize=8.5,
            ha="left",
            va="center",
        )

    ours = df[df["method"] == "UL-TransXNet"].iloc[0]
    ax.scatter(
        [ours["macs_g"]],
        [ours["mean_auc"] * 100.0],
        marker="*",
        s=520,
        facecolor="#c6292e",
        edgecolor="#7b1115",
        linewidth=1.4,
        zorder=5,
    )
    ax.annotate(
        f"UL-TransXNet\n{ours['mean_auc']*100:.1f}% mean AUC\n{ours['params_m']:.1f}M / {ours['macs_g']:.2f}G",
        xy=(ours["macs_g"], ours["mean_auc"] * 100.0),
        xytext=(ours["macs_g"] + 0.65, ours["mean_auc"] * 100.0 + 1.6),
        arrowprops={"arrowstyle": "-", "color": "#7b1115", "lw": 1.0},
        color="#8e1a1d",
        fontsize=9.5,
        weight="bold",
    )
    ax.set_xlabel("MACs (G)")
    ax.set_ylabel("Mean AUC (%)")
    ax.set_xlim(0, max(df["macs_g"]) + 0.7)
    ax.set_ylim(max(60, df["mean_auc"].min() * 100 - 3), min(98, df["mean_auc"].max() * 100 + 4))
    ax.grid(True, which="major", color="#aab0b5", alpha=0.55, linewidth=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#d8dde1", alpha=0.4, linewidth=0.4)
    ax.text(
        0.03,
        0.05,
        "Bubble area denotes parameters",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4c5962",
        bbox={"facecolor": "white", "edgecolor": "#c8d0d6", "alpha": 0.85, "pad": 2.0},
    )
    save(fig, "fig_teaser_tradeoff")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("TN5000", "BUSI", "AUL")):
        ycol = f"{dataset}_auc"
        ax.scatter(
            base["macs_g"],
            base[ycol] * 100.0,
            s=[bubble_area(v) * 0.8 for v in base["params_m"]],
            facecolor="#b9c6cf",
            edgecolor="#3a4b57",
            linewidth=1.0,
            alpha=0.95,
        )
        ax.scatter(
            [ours["macs_g"]],
            [ours[ycol] * 100.0],
            marker="*",
            s=330,
            facecolor="#c6292e",
            edgecolor="#7b1115",
            linewidth=1.2,
            zorder=4,
        )
        for row in df.itertuples(index=False):
            if row.method in {"UL-TransXNet", "Swin-T", "ConvNeXt-Tiny", "EfficientFormer-L1"}:
                ax.text(
                    row.macs_g + 0.05,
                    getattr(row, ycol) * 100.0 + 0.25,
                    row.method.replace("-Tiny", ""),
                    fontsize=7.1,
                )
        ax.set_title(dataset)
        ax.set_xlabel("MACs (G)")
        ax.grid(True, color="#c3c8cc", alpha=0.5, linewidth=0.6)
    axes[0].set_ylabel("AUC (%)")
    axes[0].set_ylim(52, 98)
    axes[0].set_xlim(0, max(df["macs_g"]) + 0.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_complexity_tradeoff_auc")
    plt.close(fig)

    print(f"Updated complexity figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
