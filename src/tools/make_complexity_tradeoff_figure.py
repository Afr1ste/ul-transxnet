from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path(r"C:\Users\Afr1ste\OneDrive\My Notes\tex\pr_ultrasound_lesion_classification")
COMPLEXITY_CSV = PROJECT_ROOT / "eval_reports" / "paper_model_complexity_table.csv"
FIG_DIR = PAPER_ROOT / "figures"


# Mean values from the current full benchmark table in sections/08_appendix.tex.
METRICS = {
    "UL-TransXNet": {"auc": [0.9483, 0.8998, 0.8767], "balacc": [0.8726, 0.7961, 0.8200]},
    "ResNet50": {"auc": [0.6531, 0.7519, 0.5481], "balacc": [0.6319, 0.6826, 0.5426]},
    "EfficientNet-B0": {"auc": [0.7856, 0.7659, 0.6099], "balacc": [0.7026, 0.7061, 0.5958]},
    "MobileNetV3-Large": {"auc": [0.8282, 0.7947, 0.7357], "balacc": [0.7626, 0.7311, 0.6611]},
    "Swin-T": {"auc": [0.9497, 0.8190, 0.8319], "balacc": [0.8782, 0.7137, 0.7783]},
    "DenseNet121": {"auc": [0.9360, 0.8154, 0.6729], "balacc": [0.8609, 0.7291, 0.6304]},
    "ConvNeXt-Tiny": {"auc": [0.9508, 0.8050, 0.8274], "balacc": [0.8794, 0.7261, 0.6898]},
    "RepViT-M1.1": {"auc": [0.8403, 0.8028, 0.6797], "balacc": [0.7603, 0.7137, 0.6320]},
    "EfficientFormer-L1": {"auc": [0.9249, 0.7917, 0.7196], "balacc": [0.8484, 0.7213, 0.6293]},
}


LABEL_OFFSETS = {
    "MobileNetV3-Large": (0.16, 0.9),
    "EfficientNet-B0": (0.18, -1.9),
    "RepViT-M1.1": (0.18, -1.5),
    "EfficientFormer-L1": (0.20, 1.1),
    "DenseNet121": (0.20, -1.8),
    "ResNet50": (0.22, -1.8),
    "ConvNeXt-Tiny": (-1.45, 1.2),
    "Swin-T": (-1.20, -2.0),
    "UL-TransXNet": (0.30, 1.0),
}


def mean_percent(values: list[float]) -> float:
    return sum(values) / len(values) * 100.0


def load_complexity() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with COMPLEXITY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method in METRICS:
                rows[method] = {
                    "params_m": float(row["params_m"]),
                    "macs_g": float(row["macs_g"]),
                }
    missing = sorted(set(METRICS) - set(rows))
    if missing:
        raise RuntimeError(f"Missing complexity rows: {missing}")
    return rows


def draw_tradeoff(metric_key: str, ylabel: str, out_stem: str) -> Path:
    complexity = load_complexity()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 1.0,
            "mathtext.fontset": "dejavuserif",
            "figure.dpi": 160,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    prior_methods = [m for m in METRICS if m != "UL-TransXNet"]
    for method in prior_methods:
        x = complexity[method]["macs_g"]
        y = mean_percent(METRICS[method][metric_key])
        params = complexity[method]["params_m"]
        ax.scatter(
            x,
            y,
            s=28 + params * 3.4,
            marker="o",
            facecolor="#9aa7b2",
            edgecolor="#2f3a43",
            linewidth=0.8,
            alpha=0.88,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get(method, (0.1, 0.8))
        ax.annotate(
            f"{method}\n{params:.1f}M",
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=7.8,
            color="#1f252b",
            arrowprops=dict(arrowstyle="-", lw=0.55, color="#7b8790", shrinkA=1.5, shrinkB=2.5),
        )

    ours = "UL-TransXNet"
    x = complexity[ours]["macs_g"]
    y = mean_percent(METRICS[ours][metric_key])
    params = complexity[ours]["params_m"]
    ax.scatter(
        x,
        y,
        s=180,
        marker="*",
        facecolor="#c9252d",
        edgecolor="#6f1116",
        linewidth=1.1,
        zorder=5,
        label="UL-TransXNet (Ours)",
    )
    dx, dy = LABEL_OFFSETS[ours]
    ax.annotate(
        f"Ours\n{params:.1f}M",
        (x, y),
        xytext=(x + dx, y + dy),
        fontsize=9.4,
        fontweight="bold",
        color="#7a1117",
        arrowprops=dict(arrowstyle="-", lw=0.8, color="#7a1117", shrinkA=1.5, shrinkB=2.5),
    )

    ax.set_xlabel("MACs (G)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.0, 7.65)
    if metric_key == "auc":
        ax.set_ylim(62.0, 94.0)
    else:
        ax.set_ylim(58.0, 84.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="#b9b9b9", linewidth=0.75, alpha=0.65)
    ax.grid(True, which="minor", color="#dedede", linewidth=0.45, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#111111")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{out_stem}.png"
    pdf_path = FIG_DIR / f"{out_stem}.pdf"
    svg_path = FIG_DIR / f"{out_stem}.svg"
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def main() -> None:
    png = draw_tradeoff("auc", "Mean AUC (%)", "fig_complexity_tradeoff_auc")
    print(f"[SAVE] {png}")


if __name__ == "__main__":
    main()
