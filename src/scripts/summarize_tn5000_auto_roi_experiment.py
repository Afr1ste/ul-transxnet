from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parent

DETECTOR_SUMMARY_DIR = PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_detector_3seed_summary_20260502_130557"
CROP_SWEEP_DIR = PROJECT_ROOT / "eval_reports" / "tn5000_roi_crop_rule_sweep_s27_20260502_141635"
LIGHT_JITTER_DIR = PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_bboxjitter_light_20260502_185435"
PREDBOX_MIX_DIR = PROJECT_ROOT / "eval_reports" / "tn5000_auto_roi_predboxmix_20260502_185435"
TRAIN_BOX_METRICS = (
    PROJECT_ROOT
    / "eval_reports"
    / "tn5000_detector_train_boxes_20260502_185435"
    / "detector_predictions"
    / "train_box_metrics.json"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def mean_std(rows: list[dict[str, str]], metric: str, where: dict[str, str] | None = None) -> tuple[float, float]:
    vals: list[float] = []
    for row in rows:
        if where and any(str(row.get(k)) != str(v) for k, v in where.items()):
            continue
        val = fnum(row.get(metric))
        if not math.isnan(val):
            vals.append(val)
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def copy_selected_rows(rows: list[dict[str, str]], predicates: list[tuple[str, dict[str, str]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for label, pred in predicates:
        for row in rows:
            if all(str(row.get(k)) == str(v) for k, v in pred.items()):
                new_row: dict[str, object] = {"row_label": label}
                new_row.update(row)
                out.append(new_row)
    return out


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "eval_reports" / f"tn5000_auto_roi_final_summary_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    detector_rows = read_csv(DETECTOR_SUMMARY_DIR / "detector_per_seed.csv")
    box_rows = read_csv(DETECTOR_SUMMARY_DIR / "box_quality_per_seed.csv")
    cls_rows = read_csv(DETECTOR_SUMMARY_DIR / "auto_roi_classification_per_seed.csv")
    crop_rows = read_csv(CROP_SWEEP_DIR / "crop_rule_test_compact.csv")
    light_rows = read_csv(LIGHT_JITTER_DIR / "auto_roi_classification_metrics.csv")
    mix_rows = read_csv(PREDBOX_MIX_DIR / "auto_roi_classification_metrics.csv")

    with TRAIN_BOX_METRICS.open("r", encoding="utf-8") as f:
        train_box = json.load(f)

    detector_summary: list[dict[str, object]] = []
    for metric in ["val_map50", "val_map50_95", "test_map50", "test_map50_95"]:
        m, s = mean_std(detector_rows, metric)
        detector_summary.append({"metric": metric, "mean": m, "std": s, "mean_std": f"{fmt(m)} +/- {fmt(s)}"})
    for metric in ["mean_iou", "median_iou", "recall_iou_0_50", "recall_iou_0_75", "no_detection_rate"]:
        m, s = mean_std(box_rows, metric, {"split": "test"})
        detector_summary.append({"metric": f"test_{metric}", "mean": m, "std": s, "mean_std": f"{fmt(m)} +/- {fmt(s)}"})

    closed_loop_summary: list[dict[str, object]] = []
    for metric in ["auc", "bal_acc", "f1_macro", "acc", "recall_0", "recall_1"]:
        m, s = mean_std(cls_rows, metric, {"split": "test"})
        closed_loop_summary.append({"metric": f"test_{metric}", "mean": m, "std": s, "mean_std": f"{fmt(m)} +/- {fmt(s)}"})

    crop_key_rows = copy_selected_rows(
        crop_rows,
        [
            ("auto_rect_expand_0.3_current", {"mode": "auto", "geometry": "rect", "expand_ratio": "0.3"}),
            ("auto_rect_expand_0.4_best_auc", {"mode": "auto", "geometry": "rect", "expand_ratio": "0.4"}),
            ("oracle_rect_expand_0.3_classifier_protocol", {"mode": "oracle", "geometry": "rect", "expand_ratio": "0.3"}),
            ("oracle_square_expand_0.3_probe", {"mode": "oracle", "geometry": "square", "expand_ratio": "0.3"}),
        ],
    )

    robust_rows: list[dict[str, object]] = []
    for exp_name, rows in [("bboxjitter_light", light_rows), ("predboxmix", mix_rows)]:
        for row in rows:
            if row.get("split") != "test":
                continue
            robust_rows.append(
                {
                    "experiment": exp_name,
                    "mode": row.get("mode"),
                    "auc": row.get("auc"),
                    "bal_acc": row.get("bal_acc"),
                    "f1_macro": row.get("f1_macro"),
                    "acc": row.get("acc"),
                    "threshold": row.get("threshold"),
                }
            )
    robust_rows.append(
        {
            "experiment": "predboxmix_train_detector_boxes",
            "mode": "train_predicted_boxes",
            "auc": "",
            "bal_acc": "",
            "f1_macro": "",
            "acc": "",
            "threshold": "",
            "n": train_box.get("n"),
            "mean_iou": train_box.get("mean_iou"),
            "median_iou": train_box.get("median_iou"),
            "recall_iou_0_50": train_box.get("recall_iou_0_50"),
            "recall_iou_0_75": train_box.get("recall_iou_0_75"),
            "no_detection_rate": train_box.get("no_detection_rate"),
        }
    )

    write_csv(out_dir / "detector_localization_summary.csv", detector_summary)
    write_csv(out_dir / "closed_loop_classification_summary.csv", closed_loop_summary)
    write_csv(out_dir / "crop_rule_key_rows.csv", crop_key_rows)
    write_csv(out_dir / "robustness_probe_summary.csv", robust_rows)

    # Keep raw per-seed rows next to the final summary to avoid hunting across directories.
    write_csv(out_dir / "detector_per_seed.csv", detector_rows)
    write_csv(out_dir / "box_quality_per_seed.csv", box_rows)
    write_csv(out_dir / "auto_roi_classification_per_seed.csv", cls_rows)

    det_map50, det_map50_std = mean_std(detector_rows, "test_map50")
    det_map5095, det_map5095_std = mean_std(detector_rows, "test_map50_95")
    iou_mean, iou_std = mean_std(box_rows, "mean_iou", {"split": "test"})
    iou75_mean, iou75_std = mean_std(box_rows, "recall_iou_0_75", {"split": "test"})
    auc_mean, auc_std = mean_std(cls_rows, "auc", {"split": "test"})
    bal_mean, bal_std = mean_std(cls_rows, "bal_acc", {"split": "test"})
    f1_mean, f1_std = mean_std(cls_rows, "f1_macro", {"split": "test"})
    acc_mean, acc_std = mean_std(cls_rows, "acc", {"split": "test"})

    summary = f"""# TN5000 automatic ROI experiment final summary

## Detector localization

- Test mAP50: {fmt(det_map50)} +/- {fmt(det_map50_std)}
- Test mAP50-95: {fmt(det_map5095)} +/- {fmt(det_map5095_std)}
- Test mean IoU: {fmt(iou_mean)} +/- {fmt(iou_std)}
- Test recall@IoU0.75: {fmt(iou75_mean)} +/- {fmt(iou75_std)}
- No-box rate: 0.0000 across all three seeds

## Closed-loop classification with detector ROI

- Test AUC: {fmt(auc_mean)} +/- {fmt(auc_std)}
- Test balanced accuracy: {fmt(bal_mean)} +/- {fmt(bal_std)}
- Test macro F1: {fmt(f1_mean)} +/- {fmt(f1_std)}
- Test accuracy: {fmt(acc_mean)} +/- {fmt(acc_std)}

## Crop-rule reading

- The current automatic ROI protocol is `auto + rect + expand=0.3`.
- In the seed-27 sweep, `auto + rect + expand=0.4` gives the highest automatic AUC, while `auto + rect + expand=0.3` gives the highest automatic balanced accuracy / macro-F1 among the compact rows.
- Oracle boxes still outperform detected boxes, so the closed-loop gap is mainly an ROI localization / crop protocol gap rather than a classifier-only issue.

## Robustness probe reading

- Light bbox jitter and predicted-box mix did not improve the automatic-ROI operating point enough to replace the original classifier.
- Keep these as negative probes or appendix diagnostics, not as the main method.

## Source directories

- Detector 3-seed summary: `{DETECTOR_SUMMARY_DIR}`
- Crop rule sweep: `{CROP_SWEEP_DIR}`
- Light bbox jitter eval: `{LIGHT_JITTER_DIR}`
- Predicted-box mix eval: `{PREDBOX_MIX_DIR}`
"""
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
