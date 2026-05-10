from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle

ROOT = Path(r"<LOCAL_THYROID_ROOT>")
PAPER_FIG_DIR = Path(r"<LOCAL_MANUSCRIPT_ROOT>\figures")
BOX_ROOT = ROOT / r"eval_reports\busi_aul_detector_box_quality_auto_20260504_165040"
CLOSED_ROOT = ROOT / r"eval_reports\busi_aul_closed_loop_auto_roi_bboxfix_20260504_182516"
OUT_STEM = PAPER_FIG_DIR / "fig_auto_roi_workflow"

DATASETS = {
    "busi": {"label": "BUSI", "fold": 0, "target_iou": 0.84, "expand": 0.30, "preferred_image_id": "test_benign_0341"},
    "aul": {"label": "AUL", "fold": 0, "target_iou": 0.70, "expand": 0.20, "preferred_image_id": "malignant_000002"},
}

COL_GT = "#2E7D32"       # restrained green
COL_PRED = "#B23A3A"     # muted red
COL_ROI = "#345E8A"      # muted blue
COL_GRAY = "#303030"


def to_float_box(row, prefix):
    return [float(row[f"{prefix}_xmin"]), float(row[f"{prefix}_ymin"]), float(row[f"{prefix}_xmax"]), float(row[f"{prefix}_ymax"])]


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))]


def expanded_roi(box, w, h, ratio):
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ew = bw * (1.0 + ratio)
    eh = bh * (1.0 + ratio)
    rx1 = cx - ew / 2.0
    ry1 = cy - eh / 2.0
    rx2 = cx + ew / 2.0
    ry2 = cy + eh / 2.0
    rx1, ry1, rx2, ry2 = clamp_box([rx1, ry1, rx2, ry2], w, h)
    return [rx1, ry1, rx2, ry2]


def draw_overlay(img, gt_box=None, pred_box=None, roi_box=None):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    def rect(box, color, width=4, dash=False):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        if not dash:
            for i in range(width):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)
        else:
            step = 18
            gap = 9
            for i in range(width):
                for x in range(x1, x2, step):
                    draw.line([(x, y1+i), (min(x+gap, x2), y1+i)], fill=color, width=1)
                    draw.line([(x, y2-i), (min(x+gap, x2), y2-i)], fill=color, width=1)
                for y in range(y1, y2, step):
                    draw.line([(x1+i, y), (x1+i, min(y+gap, y2))], fill=color, width=1)
                    draw.line([(x2-i, y), (x2-i, min(y+gap, y2))], fill=color, width=1)
    if gt_box is not None:
        rect(gt_box, COL_GT, width=3)
    if pred_box is not None:
        rect(pred_box, COL_PRED, width=4)
    if roi_box is not None:
        rect(roi_box, COL_ROI, width=4, dash=True)
    return out


def crop_roi(img, roi_box):
    w, h = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(roi_box, w, h)]
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img.crop((x1, y1, x2, y2))


def choose_case(dataset, cfg):
    box_df = pd.read_csv(BOX_ROOT / "box_quality_per_image.csv")
    pred_df = pd.read_csv(CLOSED_ROOT / dataset / "auto" / f"fold{cfg['fold']}" / "test_predictions.csv")
    sub = box_df[(box_df["dataset"] == dataset) & (box_df["fold"].astype(int) == cfg["fold"]) & (box_df["split"] == "test")].copy()
    sub = sub.merge(pred_df[["image_id", "true_label", "pred_label", "prob_class1", "threshold", "is_wrong"]], on="image_id", how="inner")
    preferred_image_id = str(cfg.get("preferred_image_id", "") or "").strip()
    if preferred_image_id:
        preferred = sub[sub["image_id"].astype(str) == preferred_image_id]
        if len(preferred) == 0:
            raise RuntimeError(f"Preferred case not found for {dataset}: {preferred_image_id}")
        return preferred.iloc[0].to_dict()

    sub = sub[(sub["no_detection"].astype(int) == 0) & (sub["is_wrong"].astype(int) == 0)].copy()
    sub["margin"] = np.abs(sub["prob_class1"].astype(float) - sub["threshold"].astype(float))
    sub["pred_area"] = (
        (sub["pred_xmax"].astype(float) - sub["pred_xmin"].astype(float))
        * (sub["pred_ymax"].astype(float) - sub["pred_ymin"].astype(float))
    )
    sub = sub[(sub["margin"] > 0.04) & (sub["pred_area"] > 10000)]
    sub["select_score"] = (
        np.abs(sub["iou"].astype(float) - cfg["target_iou"])
        - 0.02 * sub["margin"]
        - 0.0000004 * sub["pred_area"]
    )
    if len(sub) == 0:
        raise RuntimeError(f"No usable case for {dataset}")
    row = sub.sort_values("select_score").iloc[0].to_dict()
    return row


def setup_axis(ax, image, title, subtitle=None):
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#202020")
    ax.set_title(title, fontsize=10.5, pad=7, fontweight="semibold")
    if subtitle:
        ax.text(0.5, -0.075, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#303030")


def add_output_card(ax, crop_img, row):
    ax.imshow(crop_img, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#202020")
    pred = int(row["pred_label"])
    p1 = float(row["prob_class1"])
    thr = float(row["threshold"])
    ax.text(
        0.03, 0.97, f"pred={pred}, p1={p1:.2f}, thr={thr:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#A8A8A8", alpha=0.90),
        color="#202020",
    )


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
    })
    cases = []
    for ds, cfg in DATASETS.items():
        row = choose_case(ds, cfg)
        img = Image.open(row["image_path"]).convert("L")
        w, h = img.size
        gt = to_float_box(row, "gt")
        pred = to_float_box(row, "pred")
        roi = expanded_roi(pred, w, h, cfg["expand"])
        cases.append((ds, cfg, row, img, gt, pred, roi))

    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.05), dpi=300)
    col_titles = ["Original image", "Detector output", "Expanded ROI", "Classifier input / output"]

    for r, (ds, cfg, row, img, gt, pred, roi) in enumerate(cases):
        overlay_det = draw_overlay(img, gt_box=gt, pred_box=pred)
        overlay_roi = draw_overlay(img, pred_box=pred, roi_box=roi)
        crop = crop_roi(img, roi)

        setup_axis(axes[r, 0], img, col_titles[0] if r == 0 else "", cfg["label"])
        setup_axis(axes[r, 1], overlay_det, col_titles[1] if r == 0 else "", f"IoU={float(row['iou']):.2f}")
        setup_axis(axes[r, 2], overlay_roi, col_titles[2] if r == 0 else "", f"expand={int(cfg['expand']*100)}%")
        add_output_card(axes[r, 3], crop, row)
        axes[r, 3].set_title(col_titles[3] if r == 0 else "", fontsize=10.5, pad=7, fontweight="semibold")
        axes[r, 3].text(0.5, -0.075, r"$B \times 3 \times 256 \times 256$", transform=axes[r, 3].transAxes, ha="center", va="top", fontsize=8.5, color="#303030")

        axes[r, 0].text(-0.18, 0.5, cfg["label"], transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=11, fontweight="bold", color="#202020")

    legend_handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=COL_GT, linewidth=3.0, label="Reference annotation"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=COL_PRED, linewidth=3.0, label="Detector box"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=COL_ROI, linewidth=3.0, linestyle=(0, (3, 2)), label="Expanded ROI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        fontsize=12.0,
        handlelength=1.8,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.035, right=0.995, top=0.865, bottom=0.070, wspace=0.075, hspace=0.235)

    # Draw inter-panel arrows in figure coordinates.
    for r in range(2):
        row_axes = axes[r]
        y = (row_axes[0].get_position().y0 + row_axes[0].get_position().y1) / 2
        for c in range(3):
            x0 = row_axes[c].get_position().x1 + 0.01
            x1 = row_axes[c + 1].get_position().x0 - 0.01
            if x1 <= x0:
                # Tall ultrasound panels can make adjacent axes overlap after
                # aspect-ratio adjustment. Keep the visual flow left-to-right.
                mid = (row_axes[c].get_position().x1 + row_axes[c + 1].get_position().x0) / 2
                x0 = mid - 0.008
                x1 = mid + 0.008
            arrow = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                    arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
                                    color="#202020")
            fig.add_artist(arrow)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_STEM.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.03)
    selected = PAPER_FIG_DIR / "fig_auto_roi_workflow_selected_cases.txt"
    selected.write_text("\n".join([
        f"{cfg['label']}: image_id={row['image_id']}, fold={cfg['fold']}, iou={float(row['iou']):.4f}, pred={int(row['pred_label'])}, p1={float(row['prob_class1']):.4f}, thr={float(row['threshold']):.4f}, source={row['image_path']}"
        for _, cfg, row, *_ in cases
    ]) + "\n", encoding="utf-8")
    print(OUT_STEM.with_suffix('.png'))
    print(selected)
    for _, cfg, row, *_ in cases:
        print(f"{cfg['label']}: {row['image_id']} iou={float(row['iou']):.4f} pred={int(row['pred_label'])} p1={float(row['prob_class1']):.4f} thr={float(row['threshold']):.4f}")


if __name__ == "__main__":
    main()
