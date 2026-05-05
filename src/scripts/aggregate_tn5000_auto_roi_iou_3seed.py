#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate 3-seed TN5000 auto-ROI IoU bucket analyses."""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE_DIRS = [
    PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_yolo11n_3seed_s17_20260502_130557",
    PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_yolo11n_3seed_s27_20260502_130557",
    PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_yolo11n_3seed_s37_20260502_130557",
]


METRIC_KEYS = [
    "n",
    "n_benign",
    "n_malignant",
    "mean_iou",
    "median_iou",
    "mean_detector_conf",
    "auc",
    "acc",
    "bal_acc",
    "f1_macro",
    "recall_0",
    "recall_1",
    "error_rate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-dirs", nargs="+", default=[str(p) for p in DEFAULT_PIPELINE_DIRS])
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_iou_3seed_summary_20260502_130557"))
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


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return math.nan, math.nan
    return sum(vals) / len(vals), st.stdev(vals) if len(vals) > 1 else 0.0


def fmt_pair(mean: float, sd: float) -> str:
    if math.isnan(mean):
        return "nan"
    return f"{mean:.4f} +/- {sd:.4f}"


def main() -> int:
    args = parse_args()
    pipeline_dirs = [Path(p) for p in args.pipeline_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict] = []
    for pipeline_dir in pipeline_dirs:
        seed = pipeline_dir.name.split("_3seed_s", 1)[-1].split("_", 1)[0]
        metrics_path = pipeline_dir / "iou_bucket_analysis" / "auto_roi_iou_bucket_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        for row in read_csv(metrics_path):
            out = {"seed": seed, "pipeline_dir": str(pipeline_dir), **row}
            per_seed_rows.append(out)
    write_csv(out_dir / "auto_roi_iou_bucket_metrics_per_seed.csv", per_seed_rows)

    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in per_seed_rows:
        grouped.setdefault((row["split"], row["iou_bucket"]), []).append(row)

    agg_rows: List[Dict] = []
    order = ["all", "no_detection", "iou_lt_0.50", "iou_0.50_0.75", "iou_0.75_0.90", "iou_ge_0.90"]
    for split in ["val", "test"]:
        for bucket in order:
            rows = grouped.get((split, bucket), [])
            if not rows:
                continue
            agg: Dict[str, object] = {"split": split, "iou_bucket": bucket, "seeds": len(rows)}
            for key in METRIC_KEYS:
                vals = [to_float(r.get(key, "")) for r in rows]
                mean, sd = mean_std(vals)
                agg[f"{key}_mean"] = mean
                agg[f"{key}_std"] = sd
            agg_rows.append(agg)
    write_csv(out_dir / "auto_roi_iou_bucket_metrics_3seed_agg.csv", agg_rows)

    lines = [
        "# TN5000 auto ROI IoU-bucket 3-seed summary",
        "",
        "This report aggregates IoU-bucket diagnosis analyses from the three YOLO11n detector seeds.",
        "",
        "| split | IoU bucket | n | mean IoU | AUC | BalAcc | F1 | recall_0 | recall_1 | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            "| {split} | {bucket} | {n} | {miou} | {auc} | {bal} | {f1} | {r0} | {r1} | {err} |".format(
                split=row["split"],
                bucket=row["iou_bucket"],
                n=fmt_pair(float(row["n_mean"]), float(row["n_std"])),
                miou=fmt_pair(float(row["mean_iou_mean"]), float(row["mean_iou_std"])),
                auc=fmt_pair(float(row["auc_mean"]), float(row["auc_std"])),
                bal=fmt_pair(float(row["bal_acc_mean"]), float(row["bal_acc_std"])),
                f1=fmt_pair(float(row["f1_macro_mean"]), float(row["f1_macro_std"])),
                r0=fmt_pair(float(row["recall_0_mean"]), float(row["recall_0_std"])),
                r1=fmt_pair(float(row["recall_1_mean"]), float(row["recall_1_std"])),
                err=fmt_pair(float(row["error_rate_mean"]), float(row["error_rate_std"])),
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The `all` row should match the corresponding auto-ROI closed-loop classification result for each split.",
            "- Low-IoU buckets quantify how much diagnosis performance degrades when the detector crop is poor.",
            "- If high-IoU buckets still underperform oracle ROI, the residual gap is likely due to crop-context distribution shift rather than pure localization failure.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8-sig")
    print("[DONE] 3-seed IoU aggregation:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

