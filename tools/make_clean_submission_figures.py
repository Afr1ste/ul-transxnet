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
    """Hand-authored SVG layout for the architecture workflow figure."""

    width, height = 1280, 650
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
    add(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    blue, blue_fill = "#2b6f9f", "#eaf4ff"
    green, green_fill = "#3b7f4a", "#eaf7ed"
    orange, orange_fill = "#b86a2a", "#fff2e4"
    purple, purple_fill = "#7256a3", "#f1ebfb"
    gray_fill, panel_fill = "#f7f9fb", "#fbfcfd"

    # A. Teacher backbone.
    rect(28, 24, 1224, 188, fill=panel_fill, stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 55, "A", "Teacher backbone: what each stage changes")
    text(48, 83, "Input resolution is 256 x 256; spatial size halves after each stage.", size=12, color="#5b6672", anchor="start")

    def stage_box(x, y, label, resolution, channels, depth, heads, sr, fill, stroke):
        rect(x, y, 138, 86, fill=fill, stroke=stroke, sw=1.7, rx=8)
        add(f'<line x1="{x}" y1="{y + 26}" x2="{x + 138}" y2="{y + 26}" stroke="{stroke}" stroke-width="1.2" opacity="0.65"/>')
        text(x + 69, y + 14, label, size=14, weight=700, color="#1f2933")
        text(x + 69, y + 39, resolution, size=12.5, weight=700, color="#26323d")
        text(x + 69, y + 59, f"C={channels}  depth={depth}", size=11.5, color="#4d5965")
        text(x + 69, y + 76, f"heads={heads}  SR={sr}", size=11, color="#6b7480")

    y = 104
    box(54, y + 8, 100, 62, ["Expanded ROI", "3 x 256 x 256"], fill="#ffffff", stroke="#6b7785", size=12, weight=600)
    box(184, y + 8, 108, 62, ["Patch embed", "stride 4"], fill=orange_fill, stroke=orange, size=12, weight=600)
    stage_box(324, y, "Stage 1", "64 x 64", "48", "4", "1", "8", "#eef5fc", blue)
    stage_box(486, y, "Stage 2", "32 x 32", "96", "4", "2", "4", green_fill, green)
    stage_box(648, y, "Stage 3", "16 x 16", "224", "15", "4", "2", orange_fill, orange)
    stage_box(810, y, "Stage 4", "8 x 8", "448", "4", "8", "1", purple_fill, purple)
    box(992, y + 8, 70, 62, ["Global", "pool"], fill=blue_fill, stroke=blue, size=12, weight=600)
    box(1088, y + 8, 92, 62, ["1000-D", "task vector"], fill=blue_fill, stroke=blue, size=11.5, weight=600)
    box(1202, y + 8, 38, 62, ["2", "logits"], fill=blue_fill, stroke=blue, size=11.5, weight=700)
    for x1, x2 in [(156, 182), (294, 322), (464, 484), (626, 646), (788, 808), (950, 990), (1064, 1086), (1182, 1200)]:
        line(x1, 143, x2, 143)

    # B. Block-level options.
    rect(28, 236, 1224, 244, fill="#ffffff", stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 268, "B", "Representative block: where the optional modules enter")
    by = 382

    # Stage-local memory is drawn above the main path so arrows do not cross the block sequence.
    box(54, by - 74, 156, 46, ["Stage-local cache", "previous block features"], fill=green_fill, stroke=green, size=11.5, weight=600)
    box(238, by - 74, 148, 46, ["MUDD option", "dynamic dense reuse"], fill=green_fill, stroke=green, size=11.5, weight=700)
    line(212, by - 55, 236, by - 55, color=green)
    line(312, by - 32, 312, by + 14, color=green)

    block_boxes = [
        (54, by, 50, 40, "input", "#ffffff", "#79838f", 12, 600),
        (132, by, 78, 40, "DPE", orange_fill, orange, 13, 600),
        (238, by - 10, 122, 60, ["MCA option", "coordinate-aware", "refinement"], blue_fill, blue, 11, 700),
        (388, by, 70, 40, "Norm", gray_fill, "#8a939d", 12, 600),
        (486, by - 12, 148, 64, ["Local-global mixer", "DA option inside", "attention branch"], purple_fill, purple, 11.5, 700),
        (662, by, 40, 40, "+", "#ffffff", "#79838f", 18, 600),
        (730, by, 70, 40, "Norm", gray_fill, "#8a939d", 12, 600),
        (828, by - 8, 108, 56, "MS-FFN", blue_fill, blue, 13, 700),
        (964, by, 40, 40, "+", "#ffffff", "#79838f", 18, 600),
        (1032, by, 50, 40, "output", "#ffffff", "#79838f", 11, 600),
    ]
    for bx, by0, bw, bh, label, fill, stroke, size, weight in block_boxes:
        box(bx, by0, bw, bh, label, fill=fill, stroke=stroke, size=size, weight=weight)
    for x1, x2 in [(106, 130), (212, 236), (362, 386), (460, 484), (636, 660), (704, 728), (802, 826), (938, 962), (1006, 1030)]:
        line(x1, by + 20, x2, by + 20)

    box(1112, by - 48, 108, 40, ["Validation", "selection"], fill="#ffffff", stroke="#6b7785", size=11.5, weight=700)
    box(1112, by + 20, 108, 40, ["Fixed test", "reporting"], fill="#ffffff", stroke="#6b7785", size=11.5, weight=700)
    line(1084, by + 20, 1110, by - 28)
    line(1166, by - 6, 1166, by + 18)
    text(638, 456, "Options are toggled independently: MCA = coordinate recalibration; MUDD = same-stage reuse; DA = differential attention.", size=11.3, color="#596571")

    # C. Evaluation and deployment path.
    rect(28, 504, 1224, 122, fill="#fbfdfb", stroke="#d4d9df", sw=1.3, rx=12)
    title(48, 534, "C", "Evidence pathway: analysis teacher to mobile student")
    cy = 562
    path_boxes = [
        (54, cy, 124, 42, ["Frozen manifest", "and labels"], "#ffffff", "#6b7785", 11.5, 600),
        (218, cy, 130, 42, ["Teacher-family", "variants"], "#ffffff", "#6b7785", 11.5, 600),
        (388, cy, 152, 42, ["Benchmark, ablation", "and ROI robustness"], orange_fill, orange, 11.5, 600),
        (580, cy, 112, 42, ["Teacher logits", "for KD"], blue_fill, blue, 11.5, 600),
        (732, cy, 166, 42, ["EfficientFormer-L1", "+ ECA student"], green_fill, green, 11.5, 700),
        (938, cy, 116, 42, ["ONNX Runtime", "export"], "#ffffff", "#6b7785", 11.5, 600),
        (1094, cy, 126, 42, ["Two Android", "devices"], blue_fill, blue, 11.5, 700),
    ]
    for bx, by0, bw, bh, label, fill, stroke, size, weight in path_boxes:
        box(bx, by0, bw, bh, label, fill=fill, stroke=stroke, size=size, weight=weight)
    for x1, x2 in [(180, 216), (350, 386), (542, 578), (694, 730), (900, 936), (1056, 1092)]:
        line(x1, cy + 21, x2, cy + 21)
    text(464, 614, "analysis evidence", size=11, color=orange)
    text(846, 614, "deployment evidence", size=11, color=green)

    add("</svg>")
    svg_text = "\n".join(svg)
    svg_path = OUT / "fig_architecture_crop.svg"
    pdf_path = OUT / "fig_architecture_crop.pdf"
    png_path = OUT / "fig_architecture_crop.png"
    svg_path.write_text(svg_text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(png_path), output_width=2400)


def mobile_tradeoff() -> None:
    src = ROOT / "results" / "strict_20260514" / "manuscript_tables" / "table7_mobile.csv"
    runtime_src = ROOT / "results" / "strict_20260514" / "mobile" / "strict_two_device_mobile_summary_20260514.csv"
    rows: list[dict[str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))

    runtime_rows: list[dict[str, str]] = []
    with runtime_src.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            runtime_rows.append({str(k).strip('\ufeff"'): v for k, v in row.items()})
    colors = {
        "EfficientFormer-L1+ECA+KD": "#0072B2",
        "EfficientFormer-L1+KD": "#009E73",
        "EfficientFormer-L1+ECA": "#D55E00",
    }
    runtime_colors = {"CPU default": "#0072B2", "XNNPACK-4T": "#E69F00", "NNAPI": "#CC79A7"}

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7})
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.3, 2.55),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.35},
    )

    for row in rows:
        model = row["model"]
        y = float(row["android_auc"])
        xiaomi = float(row["xiaomi_ms"])
        samsung = float(row["samsung_ms"])
        ax1.scatter(xiaomi, y, marker="o", s=46, color=colors[model], edgecolor="white", linewidth=0.7)
        ax1.scatter(samsung, y, marker="s", s=46, color=colors[model], edgecolor="white", linewidth=0.7)
        ax1.plot([xiaomi, samsung], [y, y], color=colors[model], linewidth=1.1, alpha=0.45)
        label = model.replace("EfficientFormer-L1+", "")
        ax1.text(samsung + 1.8, y, label, fontsize=7.2, va="center", color=colors[model])
    ax1.set_xlabel("Hot end-to-end latency (ms)")
    ax1.set_ylabel("Android AUC")
    ax1.set_title("(a) Student accuracy-latency trade-off", fontsize=9)
    ax1.grid(axis="both", alpha=0.22, linewidth=0.7)
    ax1.set_xlim(24, 74)
    ax1.set_ylim(0.918, 0.962)
    ax1.scatter([], [], marker="o", color="#555555", label="Xiaomi")
    ax1.scatter([], [], marker="s", color="#555555", label="Samsung")
    ax1.legend(frameon=False, loc="lower right", fontsize=7)

    runtimes = ["CPU default", "XNNPACK-4T", "NNAPI"]
    x = [0, 1]
    devices = ["Xiaomi", "Samsung"]
    width_bar = 0.22
    for i, runtime in enumerate(runtimes):
        values = []
        for device_prefix in ("Xiaomi", "Samsung"):
            match = next(
                r
                for r in runtime_rows
                if r["phase"] == "hot"
                and r["device"].startswith(device_prefix)
                and r["mode"] == f"EffFormer-L1+ECA+KD{'' if runtime == 'CPU default' else ' ' + runtime}"
            )
            values.append(float(match["avg_total_ms_mean"]))
        positions = [v + (i - 1) * width_bar for v in x]
        ax2.bar(positions, values, width=width_bar, color=runtime_colors[runtime], label=runtime)
        for px, value in zip(positions, values):
            ax2.text(px, value + 5, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(devices)
    ax2.set_ylabel("Hot end-to-end latency (ms)")
    ax2.set_title("(b) Runtime backend sweep for ECA+KD", fontsize=9)
    ax2.set_ylim(0, 315)
    ax2.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")

    fig.savefig(OUT / "fig_mobile_tradeoff_crop.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_mobile_tradeoff_crop.png", bbox_inches="tight", dpi=350)
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
    mobile_tradeoff()
    roi_robustness()


if __name__ == "__main__":
    main()
