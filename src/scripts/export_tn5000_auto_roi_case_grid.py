#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export representative TN5000 auto-ROI detection/classification cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-dir", default=str(PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_yolo11n_3seed_s27_20260502_130557"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output-dir", default="")
    p.add_argument("--cases-per-group", type=int, default=4)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_box(s: str) -> Optional[Tuple[float, float, float, float]]:
    s = str(s or "").strip()
    if not s:
        return None
    parts = [float(x) for x in s.split()]
    if len(parts) != 4:
        return None
    return tuple(parts)  # type: ignore[return-value]


def merge_rows(pipeline_dir: Path, split: str) -> List[Dict]:
    boxes = {r["image_id"]: r for r in read_csv(pipeline_dir / "detector_predictions" / f"{split}_boxes.csv")}
    preds = read_csv(pipeline_dir / "classifier_auto" / f"{split}_predictions.csv")
    rows = []
    for pred in preds:
        box = boxes[pred["image_id"]]
        row = {
            **pred,
            "gt_bbox": box.get("gt_bbox", ""),
            "pred_bbox": box.get("pred_bbox", ""),
            "pred_conf": float(box.get("pred_conf", 0.0)),
            "iou_gt": float(box.get("iou_gt", 0.0)),
            "no_detection": int(str(box.get("no_detection", "0")).lower() in {"1", "true", "yes"}),
        }
        row["true_label"] = int(row["true_label"])
        row["pred_label"] = int(row["pred_label"])
        row["prob_class1"] = float(row["prob_class1"])
        row["is_wrong"] = int(row["is_wrong"])
        rows.append(row)
    return rows


def select_cases(rows: Sequence[Dict], k: int) -> List[Dict]:
    correct = [r for r in rows if r["is_wrong"] == 0]
    wrong = [r for r in rows if r["is_wrong"] == 1]
    groups = [
        ("good_box_correct", sorted([r for r in correct if r["iou_gt"] >= 0.75], key=lambda r: (-r["iou_gt"], -r["pred_conf"]))),
        ("good_box_wrong", sorted([r for r in wrong if r["iou_gt"] >= 0.75], key=lambda r: (-r["iou_gt"], -abs(r["prob_class1"] - 0.5)))),
        ("poor_box_wrong", sorted([r for r in wrong if r["iou_gt"] < 0.50], key=lambda r: (r["iou_gt"], -abs(r["prob_class1"] - 0.5)))),
        ("poor_box_correct", sorted([r for r in correct if r["iou_gt"] < 0.50], key=lambda r: (r["iou_gt"], -r["pred_conf"]))),
    ]
    selected: List[Dict] = []
    for group, group_rows in groups:
        for r in group_rows[:k]:
            rr = dict(r)
            rr["case_group"] = group
            selected.append(rr)
    return selected


def draw_cases(rows: Sequence[Dict], output_path: Path, title: str) -> None:
    n = len(rows)
    cols = 4
    rows_n = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 3.6), dpi=180)
    if rows_n == 1:
        axes = [axes]  # type: ignore[assignment]
    flat_axes = [ax for row_axes in axes for ax in row_axes]  # type: ignore[union-attr]
    for ax in flat_axes:
        ax.axis("off")
    for ax, row in zip(flat_axes, rows):
        img = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(img, cmap="gray")
        gt = parse_box(row["gt_bbox"])
        pred = parse_box(row["pred_bbox"])
        if gt is not None:
            x1, y1, x2, y2 = gt
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2ca02c", linewidth=1.6))
        if pred is not None:
            x1, y1, x2, y2 = pred
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#d62728", linewidth=1.6, linestyle="--"))
        true_name = "M" if row["true_label"] == 1 else "B"
        pred_name = "M" if row["pred_label"] == 1 else "B"
        ax.set_title(
            f"{row['case_group']}\nIoU={row['iou_gt']:.2f}, pM={row['prob_class1']:.2f}, {true_name}->{pred_name}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    pipeline_dir = Path(args.pipeline_dir)
    out_dir = Path(args.output_dir) if args.output_dir else pipeline_dir / "case_grid"
    rows = merge_rows(pipeline_dir, args.split)
    selected = select_cases(rows, args.cases_per_group)
    if not selected:
        raise RuntimeError("No cases selected.")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{args.split}_selected_cases.csv", selected)
    draw_cases(selected, out_dir / f"{args.split}_auto_roi_case_grid.png", f"TN5000 auto ROI cases ({args.split})")
    print("[DONE] case grid:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
