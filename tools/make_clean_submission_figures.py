from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg

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


def architecture() -> None:
    """Hand-authored SVG layout for the architecture workflow figure.

    The previous matplotlib layout was too close to a draft sketch.  This
    routine keeps the diagram deterministic but uses fixed publication-oriented
    SVG geometry so labels, arrows, and callouts cannot drift into each other.
    """

    width, height = 1280, 560
    svg: list[str] = []

    def add(s: str) -> None:
        svg.append(s)

    def rect(x, y, w, h, fill="#ffffff", stroke="#6b7785", sw=1.6, rx=8, cls=""):
        add(
            f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(x1, y1, x2, y2, color="#46515c", sw=1.8, marker=True, dash=""):
        marker_attr = ' marker-end="url(#arrow)"' if marker else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round"{marker_attr}{dash_attr}/>'
        )

    def text(x, y, lines, size=15, weight=400, color="#202833", anchor="middle", line_gap=18):
        if isinstance(lines, str):
            lines = [lines]
        escaped = [escape(str(t)) for t in lines]
        y0 = y - line_gap * (len(escaped) - 1) / 2
        add(
            f'<text x="{x}" y="{y0}" text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        )
        for i, item in enumerate(escaped):
            dy = 0 if i == 0 else line_gap
            add(f'<tspan x="{x}" dy="{dy}">{item}</tspan>')
        add("</text>")

    def title(x, y, letter, label):
        text(x, y, f"{letter}  {label}", size=18, weight=700, anchor="start", color="#111820")

    def box(x, y, w, h, label, fill="#ffffff", stroke="#6b7785", size=14, weight=500):
        rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.8, rx=7)
        text(x + w / 2, y + h / 2, label, size=size, weight=weight, line_gap=size + 3)

    def badge(x, y, label, fill, stroke):
        add(f'<circle cx="{x}" cy="{y}" r="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        text(x, y + 0.5, label, size=10, weight=700, color=stroke)

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    add(
        """
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="7.8" refY="4.5" orient="auto">
    <path d="M0,0 L9,4.5 L0,9 Z" fill="#46515c"/>
  </marker>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; }
  </style>
</defs>
"""
    )
    add('<rect x="0" y="0" width="1280" height="560" fill="#ffffff"/>')

    blue, blue_fill = "#2b6f9f", "#eaf4ff"
    green, green_fill = "#3b7f4a", "#eaf7ed"
    orange, orange_fill = "#b86a2a", "#fff2e4"
    purple, purple_fill = "#7256a3", "#f1ebfb"
    gray_fill, panel_fill = "#f7f9fb", "#fbfcfd"

    # A. Teacher backbone.
    rect(28, 24, 1224, 130, fill=panel_fill, stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 54, "A", "ROI teacher backbone")
    y = 82
    boxes = [
        (56, y, 96, 48, ["Expanded", "ROI crop"], "#ffffff", "#6b7785", 13, 700),
        (188, y, 88, 48, ["Patch", "embed"], orange_fill, orange, 13, 500),
        (314, y - 8, 108, 64, ["Stage 1", "C=48", "4 blocks"], "#eef5fc", "#6b7785", 13, 700),
        (456, y - 8, 108, 64, ["Stage 2", "C=96", "4 blocks"], "#edf8f1", "#6b7785", 13, 700),
        (598, y - 8, 108, 64, ["Stage 3", "C=224", "15 blocks"], "#fff5e9", "#6b7785", 13, 700),
        (740, y - 8, 108, 64, ["Stage 4", "C=448", "4 blocks"], purple_fill, "#6b7785", 13, 700),
        (892, y, 70, 48, "GAP", blue_fill, blue, 13, 500),
        (998, y, 122, 48, ["1000-D task", "projection"], blue_fill, blue, 13, 500),
        (1156, y, 88, 48, ["Binary", "head"], blue_fill, blue, 13, 700),
    ]
    for bx, by, bw, bh, label, fill, stroke, size, weight in boxes:
        box(bx, by, bw, bh, label, fill=fill, stroke=stroke, size=size, weight=weight)
    text(104, 140, "3 x 256 x 256", size=11, color="#54606d")
    for x1, x2 in [(154, 186), (278, 312), (424, 454), (566, 596), (708, 738), (850, 890), (964, 996), (1122, 1154)]:
        line(x1, 106, x2, 106)
    text(1212, 140, "teacher logits", size=11, color=blue)

    # B. Block-level options.
    rect(28, 176, 1224, 210, fill="#ffffff", stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 207, "B", "Block template and modular options")
    by = 252
    block_boxes = [
        (60, by, 48, 36, "x", "#ffffff", "#79838f", 14, 500),
        (144, by, 72, 36, "DPE", orange_fill, orange, 13, 500),
        (252, by, 74, 36, "Norm", gray_fill, "#8a939d", 13, 500),
        (362, by - 10, 126, 56, ["Local-global", "mixer"], purple_fill, purple, 13, 700),
        (526, by, 38, 36, "+", "#ffffff", "#79838f", 18, 500),
        (600, by, 74, 36, "Norm", gray_fill, "#8a939d", 13, 500),
        (710, by - 10, 112, 56, "MS-FFN", blue_fill, blue, 14, 700),
        (860, by, 38, 36, "+", "#ffffff", "#79838f", 18, 500),
        (934, by, 48, 36, "y", "#ffffff", "#79838f", 14, 500),
    ]
    for bx, by0, bw, bh, label, fill, stroke, size, weight in block_boxes:
        box(bx, by0, bw, bh, label, fill=fill, stroke=stroke, size=size, weight=weight)
    for x1, x2 in [(110, 142), (218, 250), (328, 360), (490, 524), (566, 598), (676, 708), (824, 858), (900, 932)]:
        line(x1, 270, x2, 270)
    badge(300, 244, "1", blue_fill, blue)
    badge(382, 244, "2", green_fill, green)
    badge(468, 244, "3", orange_fill, orange)

    text(430, 328, "Block-level options", size=12, weight=700, color="#3d4650")
    option_boxes = [
        (184, 342, 170, 30, "1  MCA: coordinate refinement", blue_fill, blue),
        (388, 342, 220, 30, "2  MUDD: stage-local dense reuse", green_fill, green),
        (642, 342, 190, 30, "3  DA: differential attention", orange_fill, orange),
    ]
    for bx, by0, bw, bh, label, fill, stroke in option_boxes:
        box(bx, by0, bw, bh, label, fill=fill, stroke=stroke, size=11, weight=500)

    box(1028, 228, 172, 74, ["Validation-fixed", "variant analysis"], fill="#ffffff", stroke="#6b7785", size=13, weight=600)
    line(984, 270, 1026, 270)
    text(1114, 330, ["Modules are toggled", "before test evaluation"], size=11, color="#596571", line_gap=14)

    # C. Evaluation and deployment path.
    rect(28, 408, 1224, 128, fill="#fbfdfb", stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 439, "C", "Evaluation and deployment pathway")
    cy = 470
    path_boxes = [
        (58, cy, 108, 42, ["Teacher", "variants"], "#ffffff", "#6b7785", 12, 500),
        (206, cy, 148, 42, ["ROI robustness", "GT / noisy / detector"], orange_fill, orange, 12, 500),
        (396, cy, 124, 42, ["Case-level", "predictions"], "#ffffff", "#6b7785", 12, 500),
        (562, cy, 104, 42, ["KD loss", "CE + KL"], blue_fill, blue, 12, 500),
        (708, cy, 176, 42, ["EfficientFormer-L1", "+ ECA student"], green_fill, green, 12, 700),
        (926, cy, 132, 42, ["ONNX / TFLite", "export"], "#ffffff", "#6b7785", 12, 500),
        (1100, cy, 104, 42, ["Android", "latency"], blue_fill, blue, 12, 500),
    ]
    for bx, by0, bw, bh, label, fill, stroke, size, weight in path_boxes:
        box(bx, by0, bw, bh, label, fill=fill, stroke=stroke, size=size, weight=weight)
    for x1, x2 in [(168, 204), (356, 394), (522, 560), (668, 706), (886, 924), (1060, 1098)]:
        line(x1, 491, x2, 491)
    text(280, 524, "analysis evidence", size=11, color=orange)
    text(796, 524, "deployment evidence", size=11, color=green)

    add("</svg>")
    svg_text = "\n".join(svg)
    svg_path = OUT / "fig_architecture_crop.svg"
    pdf_path = OUT / "fig_architecture_crop.pdf"
    png_path = OUT / "fig_architecture_crop.png"
    svg_path.write_text(svg_text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(png_path), output_width=2400)


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
