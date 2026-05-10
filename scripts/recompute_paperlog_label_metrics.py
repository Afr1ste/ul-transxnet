"""Recompute manuscript metrics with the frozen paper-log label snapshot.

This script exists to prevent a subtle but severe provenance error: several
historical prediction CSVs contain labels from older dataset snapshots.  The
paper must report metrics against the reconstructed paper-log labels, not
against the labels embedded in those CSVs.

The script intentionally recomputes metrics from predictions plus
``paper_log_case_labels.csv`` and writes both summary tables and an explicit
label-mismatch audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THYROID_ROOT = REPO_ROOT.parent / "Thyroid"
DEFAULT_LABELS = (
    DEFAULT_THYROID_ROOT
    / "eval_reports"
    / "paper_log_label_reconstruction_20260505"
    / "paper_log_case_labels.csv"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "results"
    / "provenance_release_20260510"
    / "predictions"
    / "recomputed_paperlog_labels"
)

DATASET_ORDER = ["TN5000", "BUSI", "AUL"]
METRIC_ORDER = ["auc", "bal_acc", "f1", "acc"]


@dataclass(frozen=True)
class Source:
    table_group: str
    dataset: str
    method: str
    log_rel: str
    display_name: str
    params_m: float | None = None
    macs_g: float | None = None
    cuda_ms: float | None = None


SOURCES: list[Source] = [
    Source(
        "ul",
        "TN5000",
        "UL-TransXNet",
        "tn5000_ggg_mca_enabled_3seed_logs/20260426_093728",
        "Ours-GGG-MCAON",
    ),
    Source(
        "ul",
        "BUSI",
        "UL-TransXNet",
        "busi_ggg_mca_clean_5fold_safe_logs/20260426_165332",
        "TransXNet-GGG",
    ),
    Source(
        "ul",
        "AUL",
        "UL-TransXNet",
        "aul_ggg_mca_clean_5fold_safe_logs/20260426_200618",
        "TransXNet-GGG",
    ),
    Source(
        "original_transxnet",
        "TN5000",
        "TransXNet",
        "tn5000_p0_structure_current_3seed_logs/20260429_030247",
        "TransXNet",
    ),
    Source(
        "original_transxnet",
        "BUSI",
        "TransXNet",
        "busi_p0_structure_clean_5fold_logs/20260425_034843",
        "TransXNet",
    ),
    Source(
        "original_transxnet",
        "AUL",
        "TransXNet",
        "aul_p0_structure_clean_5fold_logs/20260425_030033",
        "TransXNet",
    ),
    Source(
        "strong_baseline",
        "TN5000",
        "ConvNeXt-Tiny",
        "tn5000_compare_extra4models_3seed_logs/20260421_222342_merged_complete",
        "ConvNeXt-Tiny",
    ),
    Source(
        "strong_baseline",
        "BUSI",
        "Swin-T",
        "busi_compare_5models_5fold_logs/20260403_083238",
        "Swin-T",
    ),
    Source(
        "strong_baseline",
        "AUL",
        "Swin-T",
        "aul_roi_compare_5models_5fold_logs/20260404_235703",
        "Swin-T",
    ),
]


COMPLEXITY_SOURCES: list[Source] = [
    Source(
        "complexity",
        "TN5000",
        "UL-TransXNet",
        "tn5000_ggg_mca_enabled_3seed_logs/20260426_093728",
        "Ours-GGG-MCAON",
        14.4,
        2.36,
        30.23,
    ),
    Source(
        "complexity",
        "BUSI",
        "UL-TransXNet",
        "busi_ggg_mca_clean_5fold_safe_logs/20260426_165332",
        "TransXNet-GGG",
        14.4,
        2.36,
        30.23,
    ),
    Source(
        "complexity",
        "AUL",
        "UL-TransXNet",
        "aul_ggg_mca_clean_5fold_safe_logs/20260426_200618",
        "TransXNet-GGG",
        14.4,
        2.36,
        30.23,
    ),
    Source(
        "complexity",
        "TN5000",
        "MobileNetV3-Large",
        "tn5000_compare_5models_3seed_logs/20260402_192605",
        "MobileNetV3-Large",
        4.9,
        0.27,
        2.58,
    ),
    Source(
        "complexity",
        "BUSI",
        "MobileNetV3-Large",
        "busi_compare_5models_5fold_logs/20260403_083238",
        "MobileNetV3-Large",
        4.9,
        0.27,
        2.58,
    ),
    Source(
        "complexity",
        "AUL",
        "MobileNetV3-Large",
        "aul_roi_compare_5models_5fold_logs/20260404_235703",
        "MobileNetV3-Large",
        4.9,
        0.27,
        2.58,
    ),
    Source(
        "complexity",
        "TN5000",
        "EfficientNet-B0",
        "tn5000_compare_5models_3seed_logs/20260402_192605",
        "EfficientNet-B0",
        4.7,
        0.51,
        3.78,
    ),
    Source(
        "complexity",
        "BUSI",
        "EfficientNet-B0",
        "busi_compare_5models_5fold_logs/20260403_083238",
        "EfficientNet-B0",
        4.7,
        0.51,
        3.78,
    ),
    Source(
        "complexity",
        "AUL",
        "EfficientNet-B0",
        "aul_roi_compare_5models_5fold_logs/20260404_235703",
        "EfficientNet-B0",
        4.7,
        0.51,
        3.78,
    ),
    Source(
        "complexity",
        "TN5000",
        "ResNet50",
        "tn5000_compare_5models_3seed_logs/20260402_192605",
        "ResNet50",
        24.6,
        5.40,
        3.97,
    ),
    Source(
        "complexity",
        "BUSI",
        "ResNet50",
        "busi_compare_5models_5fold_logs/20260403_083238",
        "ResNet50",
        24.6,
        5.40,
        3.97,
    ),
    Source(
        "complexity",
        "AUL",
        "ResNet50",
        "aul_roi_compare_5models_5fold_logs/20260404_235703",
        "ResNet50",
        24.6,
        5.40,
        3.97,
    ),
    Source(
        "complexity",
        "TN5000",
        "Swin-T",
        "tn5000_compare_5models_3seed_logs/20260402_192605",
        "Swin-T",
        27.9,
        7.10,
        6.32,
    ),
    Source(
        "complexity",
        "BUSI",
        "Swin-T",
        "busi_compare_5models_5fold_logs/20260403_083238",
        "Swin-T",
        27.9,
        7.10,
        6.32,
    ),
    Source(
        "complexity",
        "AUL",
        "Swin-T",
        "aul_roi_compare_5models_5fold_logs/20260404_235703",
        "Swin-T",
        27.9,
        7.10,
        6.32,
    ),
    Source(
        "complexity",
        "TN5000",
        "ConvNeXt-Tiny",
        "tn5000_compare_extra4models_3seed_logs/20260421_222342_merged_complete",
        "ConvNeXt-Tiny",
        28.2,
        5.82,
        3.30,
    ),
    Source(
        "complexity",
        "BUSI",
        "ConvNeXt-Tiny",
        "busi_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "ConvNeXt-Tiny",
        28.2,
        5.82,
        3.30,
    ),
    Source(
        "complexity",
        "AUL",
        "ConvNeXt-Tiny",
        "aul_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "ConvNeXt-Tiny",
        28.2,
        5.82,
        3.30,
    ),
    Source(
        "complexity",
        "TN5000",
        "DenseNet121",
        "tn5000_compare_extra4models_3seed_logs/20260421_222342_merged_complete",
        "DenseNet121",
        7.5,
        3.70,
        5.72,
    ),
    Source(
        "complexity",
        "BUSI",
        "DenseNet121",
        "busi_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "DenseNet121",
        7.5,
        3.70,
        5.72,
    ),
    Source(
        "complexity",
        "AUL",
        "DenseNet121",
        "aul_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "DenseNet121",
        7.5,
        3.70,
        5.72,
    ),
    Source(
        "complexity",
        "TN5000",
        "EfficientFormer-L1",
        "tn5000_compare_extra4models_3seed_logs/20260421_222342_merged_complete",
        "EfficientFormer-L1",
        11.6,
        1.31,
        5.51,
    ),
    Source(
        "complexity",
        "BUSI",
        "EfficientFormer-L1",
        "busi_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "EfficientFormer-L1",
        11.6,
        1.31,
        5.51,
    ),
    Source(
        "complexity",
        "AUL",
        "EfficientFormer-L1",
        "aul_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "EfficientFormer-L1",
        11.6,
        1.31,
        5.51,
    ),
    Source(
        "complexity",
        "TN5000",
        "RepViT-M1.1",
        "tn5000_compare_extra4models_3seed_logs/20260421_222342_merged_complete",
        "RepViT-M1.1",
        8.0,
        1.78,
        2.61,
    ),
    Source(
        "complexity",
        "BUSI",
        "RepViT-M1.1",
        "busi_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "RepViT-M1.1",
        8.0,
        1.78,
        2.61,
    ),
    Source(
        "complexity",
        "AUL",
        "RepViT-M1.1",
        "aul_compare_extra4models_5fold_logs/20260421_222342_merged_complete",
        "RepViT-M1.1",
        8.0,
        1.78,
        2.61,
    ),
]


def fmt_mean_std(mu: float, sd: float, digits: int = 4) -> str:
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def load_labels(path: Path) -> dict[tuple[str, str], int]:
    df = pd.read_csv(path, dtype=str)
    label_col = "label" if "label" in df.columns else "true_label"
    required = {"dataset", "image_id", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    labels: dict[tuple[str, str], int] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        key = (str(row_dict["dataset"]), str(row_dict["image_id"]))
        label = int(row_dict[label_col])
        if key in labels and labels[key] != label:
            raise ValueError(f"conflicting labels for {key}")
        labels[key] = label
    return labels


def read_run_rows(metrics_csv: Path, display_name: str) -> list[dict]:
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)
    with metrics_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = [
        r
        for r in rows
        if r.get("display_name") == display_name
        or r.get("display_name_cfg") == display_name
        or r.get("display_name_runtime") == display_name
    ]
    if not out:
        raise ValueError(f"No rows for display_name={display_name!r} in {metrics_csv}")
    return out


def find_prediction_csv(row: dict, log_dir: Path) -> Path:
    candidates: list[Path] = []
    for key in ("pred_csv", "prediction_csv", "test_predictions_csv"):
        value = row.get(key)
        if value:
            candidates.append(Path(value))
    output_dir = row.get("output_dir") or row.get("run_dir") or row.get("log_dir")
    if output_dir:
        od = Path(output_dir)
        candidates.extend(
            [
                od / "test_predictions.csv",
                od / "test_ensemble_predictions.csv",
                od / "predictions" / "test_predictions.csv",
            ]
        )
    for subdir_key in ("run_name", "name", "fold", "seed"):
        value = row.get(subdir_key)
        if value:
            candidates.extend(
                [
                    log_dir / str(value) / "test_predictions.csv",
                    log_dir / str(value) / "test_ensemble_predictions.csv",
                ]
            )
    # Historical rows usually contain only output_dir; this fallback keeps the
    # script useful when all_runs_metrics.csv was manually edited.
    if "output_dir" not in row:
        candidates.extend(log_dir.glob("**/test_predictions.csv"))
    checked: list[str] = []
    for cand in candidates:
        p = cand
        if not p.is_absolute():
            p = (log_dir / p).resolve()
        checked.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find test prediction CSV. Checked:\n" + "\n".join(checked[:20])
    )


def prob_column(df: pd.DataFrame) -> str:
    for col in (
        "prob_class1",
        "malignant_prob",
        "prob_malignant",
        "p_malignant",
        "score",
        "y_score",
        "prob_1",
    ):
        if col in df.columns:
            return col
    prob_cols = [c for c in df.columns if c.startswith("prob")]
    if prob_cols:
        return prob_cols[-1]
    raise ValueError(f"No probability column found in {list(df.columns)}")


def id_column(df: pd.DataFrame) -> str:
    for col in ("image_id", "case_id", "sample_id", "filename", "file_name", "path"):
        if col in df.columns:
            return col
    raise ValueError(f"No image id column found in {list(df.columns)}")


def label_column(df: pd.DataFrame) -> str | None:
    for col in ("true_label", "label", "target", "y_true"):
        if col in df.columns:
            return col
    return None


def pred_column(df: pd.DataFrame) -> str | None:
    for col in ("pred_label", "prediction", "pred", "y_pred"):
        if col in df.columns:
            return col
    return None


def normalize_id(value: object) -> str:
    text = str(value).replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    return Path(name).stem


def threshold_from_scores(y_true: list[int], prob: list[float]) -> float:
    # Use the threshold that maximizes balanced accuracy on the evaluated split.
    # This mirrors the historical operating-point reporting convention; AUC is
    # unaffected by this choice.
    pairs = sorted(zip((float(x) for x in prob), y_true), key=lambda x: x[0], reverse=True)
    if not pairs:
        return 0.5
    total_pos = sum(1 for _, y in pairs if y == 1)
    total_neg = len(pairs) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    best_thr = pairs[0][0]
    best_score = -1.0
    i = 0
    n = len(pairs)
    while i < n:
        score_value = pairs[i][0]
        while i < n and pairs[i][0] == score_value:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        tn = total_neg - fp
        score = 0.5 * ((tp / total_pos) + (tn / total_neg))
        if score > best_score:
            best_score = score
            best_thr = score_value
    return best_thr


def metrics_from_arrays(
    y_true: list[int],
    prob: list[float],
    threshold: float | None = None,
    pred: list[int] | None = None,
) -> dict[str, float]:
    if pred is None and threshold is None:
        threshold = threshold_from_scores(y_true, prob)
    if pred is None:
        pred = [1 if p >= float(threshold) else 0 for p in prob]
    return {
        "auc": roc_auc_score(y_true, prob),
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "acc": accuracy_score(y_true, pred),
        "threshold": float(threshold) if threshold is not None else math.nan,
    }


def ece_score(y_true: list[int], prob: list[float], n_bins: int = 10) -> float:
    conf = [max(p, 1.0 - p) for p in prob]
    pred = [1 if p >= 0.5 else 0 for p in prob]
    correct = [1 if a == b else 0 for a, b in zip(pred, y_true)]
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [j for j, c in enumerate(conf) if (lo <= c < hi) or (i == n_bins - 1 and c == hi)]
        if not idx:
            continue
        avg_conf = mean(conf[j] for j in idx)
        avg_acc = mean(correct[j] for j in idx)
        ece += (len(idx) / n) * abs(avg_acc - avg_conf)
    return ece


def diagnostic_metrics(y_true: list[int], prob: list[float]) -> dict[str, float]:
    threshold = threshold_from_scores(y_true, prob)
    pred = [1 if p >= threshold else 0 for p in prob]
    tp = sum(1 for y, z in zip(y_true, pred) if y == 1 and z == 1)
    tn = sum(1 for y, z in zip(y_true, pred) if y == 0 and z == 0)
    fp = sum(1 for y, z in zip(y_true, pred) if y == 0 and z == 1)
    fn = sum(1 for y, z in zip(y_true, pred) if y == 1 and z == 0)
    sens = tp / (tp + fn) if tp + fn else math.nan
    spec = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    base = metrics_from_arrays(y_true, prob, threshold)
    base.update(
        {
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "ece": ece_score(y_true, prob),
            "brier": brier_score_loss(y_true, prob),
        }
    )
    return base


def bootstrap_ci(
    y_true: list[int],
    prob: list[float],
    metric: str,
    seed: int = 20260510,
    n_boot: int = 1000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y_true[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        pp = [prob[i] for i in idx]
        try:
            vals.append(diagnostic_metrics(yy, pp)[metric])
        except Exception:
            continue
    vals.sort()
    if not vals:
        return math.nan, math.nan
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi


def recompute_prediction_file(
    dataset: str,
    pred_csv: Path,
    labels: dict[tuple[str, str], int],
) -> tuple[dict[str, float], dict, pd.DataFrame]:
    df = pd.read_csv(pred_csv, dtype=str)
    pcol = prob_column(df)
    icol = id_column(df)
    lcol = label_column(df)
    pred_col = pred_column(df)
    image_ids = [normalize_id(v) for v in df[icol].tolist()]
    y_true: list[int] = []
    missing: list[str] = []
    for image_id in image_ids:
        key = (dataset, image_id)
        if key not in labels:
            missing.append(image_id)
        else:
            y_true.append(labels[key])
    if missing:
        sample = ", ".join(missing[:10])
        raise KeyError(f"{pred_csv} has {len(missing)} ids missing from snapshot: {sample}")
    prob = [float(v) for v in df[pcol].tolist()]
    pred = [int(v) for v in df[pred_col].tolist()] if pred_col is not None else None
    threshold = None
    if "threshold" in df.columns:
        thresh_vals = [float(v) for v in df["threshold"].dropna().tolist()]
        if thresh_vals:
            threshold = thresh_vals[0]
    metrics = metrics_from_arrays(y_true, prob, threshold=threshold, pred=pred)
    old_counts: dict[str, int] = {}
    changed = 0
    if lcol is not None:
        old = [int(v) for v in df[lcol].tolist()]
        changed = sum(1 for a, b in zip(old, y_true) if a != b)
        old_counts = {"old_benign": old.count(0), "old_malignant": old.count(1)}
    audit = {
        "prediction_csv": str(pred_csv),
        "n": len(y_true),
        "changed_labels": changed,
        "new_benign": y_true.count(0),
        "new_malignant": y_true.count(1),
        **old_counts,
    }
    out_df = pd.DataFrame(
        {
            "dataset": dataset,
            "image_id": image_ids,
            "label": y_true,
            "prob_class1": prob,
        }
    )
    return metrics, audit, out_df


def summarize_run_metrics(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    summary: dict[str, float] = {}
    for metric in METRIC_ORDER:
        vals = [float(r[metric]) for r in rows]
        summary[f"{metric}_mean"] = mean(vals)
        summary[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    summary["n_runs"] = len(rows)
    return summary


def add_source_rows(
    source: Source,
    thyroid_root: Path,
    labels: dict[tuple[str, str], int],
    run_rows: list[dict],
    audit_rows: list[dict],
    pred_frames: dict[tuple[str, str], list[pd.DataFrame]],
) -> None:
    log_dir = thyroid_root / source.log_rel
    metrics_csv = log_dir / "all_runs_metrics.csv"
    rows = read_run_rows(metrics_csv, source.display_name)
    for i, row in enumerate(rows, start=1):
        pred_csv = find_prediction_csv(row, log_dir)
        metrics, audit, pred_frame = recompute_prediction_file(source.dataset, pred_csv, labels)
        run_id = row.get("seed") or row.get("fold") or row.get("run_name") or str(i)
        out = {
            "table_group": source.table_group,
            "dataset": source.dataset,
            "method": source.method,
            "display_name": source.display_name,
            "run_index": i,
            "run_id": run_id,
            "source_log": str(log_dir),
            **metrics,
        }
        run_rows.append(out)
        audit_rows.append(
            {
                "table_group": source.table_group,
                "dataset": source.dataset,
                "method": source.method,
                "run_index": i,
                "run_id": run_id,
                "source_log": str(log_dir),
                **audit,
            }
        )
        pred_frames[(source.dataset, source.method)].append(pred_frame)


def aggregate_predictions(frames: list[pd.DataFrame]) -> tuple[list[int], list[float]]:
    if not frames:
        raise ValueError("No prediction frames")
    merged = frames[0][["image_id", "label", "prob_class1"]].rename(
        columns={"prob_class1": "prob_0"}
    )
    for i, df in enumerate(frames[1:], start=1):
        merged = merged.merge(
            df[["image_id", "label", "prob_class1"]].rename(
                columns={"label": f"label_{i}", "prob_class1": f"prob_{i}"}
            ),
            on="image_id",
            how="inner",
        )
        if not (merged["label"] == merged[f"label_{i}"]).all():
            raise ValueError("Inconsistent labels while averaging predictions")
    prob_cols = [c for c in merged.columns if c.startswith("prob_")]
    merged["prob_mean"] = merged[prob_cols].mean(axis=1)
    return merged["label"].astype(int).tolist(), merged["prob_mean"].astype(float).tolist()


def exact_signflip_p(deltas: list[float]) -> float:
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    pos = sum(1 for d in nonzero if d > 0)
    # Two-sided exact binomial sign test under p=0.5.
    from math import comb

    lower = sum(comb(n, k) for k in range(0, pos + 1)) / (2**n)
    upper = sum(comb(n, k) for k in range(pos, n + 1)) / (2**n)
    return min(1.0, 2.0 * min(lower, upper))


def write_main_tables(
    out_dir: Path,
    run_rows: list[dict],
    pred_frames: dict[tuple[str, str], list[pd.DataFrame]],
) -> dict:
    df = pd.DataFrame(run_rows)
    records: dict[str, object] = {}

    main_pairs = {
        ("TN5000", "UL-TransXNet"),
        ("TN5000", "ConvNeXt-Tiny"),
        ("BUSI", "UL-TransXNet"),
        ("BUSI", "Swin-T"),
        ("AUL", "UL-TransXNet"),
        ("AUL", "Swin-T"),
    }
    main_rows: list[dict] = []
    for dataset, method in sorted(main_pairs, key=lambda x: (DATASET_ORDER.index(x[0]), x[1])):
        sub = df[(df.dataset == dataset) & (df.method == method)]
        summary = summarize_run_metrics(sub.to_dict("records"))
        main_rows.append({"dataset": dataset, "method": method, **summary})
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out_dir / "main_benchmark_table_recomputed.csv", index=False)

    with (out_dir / "main_benchmark_rows.tex").open("w", encoding="utf-8") as f:
        for row in main_rows:
            bold = row["method"] == "UL-TransXNet" and row["dataset"] != "AUL"
            parts = [
                row["dataset"],
                row["method"],
                fmt_mean_std(row["auc_mean"], row["auc_std"]),
                fmt_mean_std(row["bal_acc_mean"], row["bal_acc_std"]),
                fmt_mean_std(row["f1_mean"], row["f1_std"]),
                fmt_mean_std(row["acc_mean"], row["acc_std"]),
            ]
            if bold:
                parts[2:] = [f"\\textbf{{{p}}}" for p in parts[2:]]
            if row["dataset"] == "AUL" and row["method"] == "Swin-T":
                parts[2:] = [f"\\textbf{{{p}}}" for p in parts[2:]]
            f.write(" & ".join(parts) + " \\\\\n")

    delta_rows: list[dict] = []
    with (out_dir / "original_transxnet_delta_rows.tex").open("w", encoding="utf-8") as f:
        for dataset in DATASET_ORDER:
            tr = summarize_run_metrics(
                df[(df.dataset == dataset) & (df.method == "TransXNet")].to_dict("records")
            )
            ul = summarize_run_metrics(
                df[(df.dataset == dataset) & (df.method == "UL-TransXNet")].to_dict("records")
            )
            row = {
                "dataset": dataset,
                "transxnet_auc": tr["auc_mean"],
                "transxnet_bal_acc": tr["bal_acc_mean"],
                "ul_auc": ul["auc_mean"],
                "ul_bal_acc": ul["bal_acc_mean"],
                "delta_auc": ul["auc_mean"] - tr["auc_mean"],
                "delta_bal_acc": ul["bal_acc_mean"] - tr["bal_acc_mean"],
            }
            delta_rows.append(row)
            f.write(
                f"{dataset} & {fmt4(row['transxnet_auc'])} & {fmt4(row['transxnet_bal_acc'])} "
                f"& {fmt4(row['ul_auc'])} & {fmt4(row['ul_bal_acc'])} "
                f"& {row['delta_auc']:+.4f} & {row['delta_bal_acc']:+.4f} \\\\\n"
            )
    pd.DataFrame(delta_rows).to_csv(out_dir / "original_transxnet_delta_recomputed.csv", index=False)

    # Case-level diagnostics for seed/fold-averaged UL-TransXNet probabilities.
    diag_rows: list[dict] = []
    cal_rows: list[dict] = []
    for dataset in DATASET_ORDER:
        y, p = aggregate_predictions(pred_frames[(dataset, "UL-TransXNet")])
        diag = diagnostic_metrics(y, p)
        ci_auc = bootstrap_ci(y, p, "auc")
        ci_bal = bootstrap_ci(y, p, "bal_acc")
        row = {
            "dataset": dataset,
            "n_cases": len(y),
            "auc": diag["auc"],
            "auc_ci_low": ci_auc[0],
            "auc_ci_high": ci_auc[1],
            "bal_acc": diag["bal_acc"],
            "bal_acc_ci_low": ci_bal[0],
            "bal_acc_ci_high": ci_bal[1],
            "sensitivity": diag["sensitivity"],
            "specificity": diag["specificity"],
            "ppv": diag["ppv"],
            "npv": diag["npv"],
            "threshold": diag["threshold"],
            "ece": diag["ece"],
            "brier": diag["brier"],
        }
        diag_rows.append(row)
        cal_rows.append(
            {
                "dataset": dataset,
                "ece": diag["ece"],
                "brier": diag["brier"],
                "auc_ci": f"{diag['auc']:.3f} [{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]",
                "bal_acc_ci": f"{diag['bal_acc']:.3f} [{ci_bal[0]:.3f}, {ci_bal[1]:.3f}]",
            }
        )
    pd.DataFrame(diag_rows).to_csv(out_dir / "case_level_ul_diagnostics_recomputed.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_uncertainty_recomputed.csv", index=False)

    with (out_dir / "case_level_diagnostic_rows.tex").open("w", encoding="utf-8") as f:
        for row in diag_rows:
            f.write(
                f"{row['dataset']} & {row['n_cases']} & "
                f"{row['auc']:.3f} [{row['auc_ci_low']:.3f}, {row['auc_ci_high']:.3f}] & "
                f"{row['bal_acc']:.3f} [{row['bal_acc_ci_low']:.3f}, {row['bal_acc_ci_high']:.3f}] & "
                f"{row['sensitivity']:.3f} & {row['specificity']:.3f} & "
                f"{row['ppv']:.3f} & {row['npv']:.3f} & {row['threshold']:.3f} \\\\\n"
            )

    with (out_dir / "calibration_uncertainty_rows.tex").open("w", encoding="utf-8") as f:
        for row in cal_rows:
            f.write(
                f"{row['dataset']} & {row['ece']:.4f} & {row['brier']:.4f} & "
                f"{row['auc_ci']} & {row['bal_acc_ci']} \\\\\n"
            )

    # Paired sign-flip checks for UL-TransXNet vs original TransXNet.
    paired_rows: list[dict] = []
    for dataset in DATASET_ORDER:
        ul = (
            df[(df.dataset == dataset) & (df.method == "UL-TransXNet")]
            .sort_values("run_index")
            .reset_index(drop=True)
        )
        tr = (
            df[(df.dataset == dataset) & (df.method == "TransXNet")]
            .sort_values("run_index")
            .reset_index(drop=True)
        )
        n = min(len(ul), len(tr))
        for metric in ("auc", "bal_acc"):
            deltas = [float(ul.loc[i, metric]) - float(tr.loc[i, metric]) for i in range(n)]
            paired_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_delta": mean(deltas),
                    "p_signflip": exact_signflip_p(deltas),
                    "all_deltas": ";".join(f"{d:.6f}" for d in deltas),
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_delta_recomputed.csv", index=False)
    with (out_dir / "paired_delta_rows.tex").open("w", encoding="utf-8") as f:
        for row in paired_rows:
            f.write(
                f"{row['dataset']} & {row['metric'].replace('_', ' ')} & "
                f"{row['n_pairs']} & {row['mean_delta']:+.4f} & {row['p_signflip']:.4f} \\\\\n"
            )

    records["main_benchmark"] = main_rows
    records["original_delta"] = delta_rows
    records["case_level"] = diag_rows
    records["calibration"] = cal_rows
    records["paired"] = paired_rows
    return records


def write_complexity_table(
    out_dir: Path,
    thyroid_root: Path,
    labels: dict[tuple[str, str], int],
) -> list[dict]:
    rows: list[dict] = []
    audits: list[dict] = []
    frames: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for source in COMPLEXITY_SOURCES:
        add_source_rows(source, thyroid_root, labels, rows, audits, frames)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(out_dir / "complexity_run_level_recomputed.csv", index=False)
    pd.DataFrame(audits).to_csv(out_dir / "complexity_label_mismatch_audit.csv", index=False)

    method_rows: list[dict] = []
    for method in sorted(run_df.method.unique()):
        per_dataset = {}
        for dataset in DATASET_ORDER:
            sub = run_df[(run_df.method == method) & (run_df.dataset == dataset)]
            summary = summarize_run_metrics(sub.to_dict("records"))
            per_dataset[dataset] = summary
        sample_source = next(s for s in COMPLEXITY_SOURCES if s.method == method)
        method_rows.append(
            {
                "method": method,
                "params_m": sample_source.params_m,
                "macs_g": sample_source.macs_g,
                "cuda_ms": sample_source.cuda_ms,
                "mean_auc": mean(per_dataset[d]["auc_mean"] for d in DATASET_ORDER),
                "mean_bal_acc": mean(per_dataset[d]["bal_acc_mean"] for d in DATASET_ORDER),
                **{f"{d}_auc": per_dataset[d]["auc_mean"] for d in DATASET_ORDER},
                **{f"{d}_bal_acc": per_dataset[d]["bal_acc_mean"] for d in DATASET_ORDER},
            }
        )
    method_rows.sort(key=lambda r: r["mean_auc"], reverse=True)
    pd.DataFrame(method_rows).to_csv(out_dir / "full_complexity_pool_recomputed.csv", index=False)
    with (out_dir / "full_complexity_pool_rows.tex").open("w", encoding="utf-8") as f:
        for row in method_rows:
            f.write(
                f"{row['method']} & {row['params_m']:.1f} & {row['macs_g']:.2f} & "
                f"{row['cuda_ms']:.2f} & {row['TN5000_auc']:.4f} & "
                f"{row['BUSI_auc']:.4f} & {row['AUL_auc']:.4f} & "
                f"{row['mean_auc']:.4f} & {row['mean_bal_acc']:.4f} \\\\\n"
            )
    return method_rows


def write_auto_roi_table(
    out_dir: Path,
    labels: dict[tuple[str, str], int],
) -> list[dict]:
    root = REPO_ROOT / "results" / "busi_aul_closed_loop_auto_roi_bboxfix_20260504_182516"
    rows: list[dict] = []
    audits: list[dict] = []
    for dataset in ("BUSI", "AUL"):
        dataset_dir = dataset.lower()
        for input_name, input_dir in (
            ("oracle_roi", "oracle"),
            ("auto_roi", "auto"),
            ("full_image", "full"),
        ):
            pred_files = sorted((root / dataset_dir / input_dir).glob("fold*/test_predictions.csv"))
            if not pred_files:
                continue
            run_metrics: list[dict] = []
            frames: list[pd.DataFrame] = []
            for i, pred_file in enumerate(pred_files, start=1):
                metrics, audit, frame = recompute_prediction_file(dataset, pred_file, labels)
                run_metrics.append(metrics)
                frames.append(frame)
                audits.append(
                    {
                        "dataset": dataset,
                        "input": input_name,
                        "run_index": i,
                        **audit,
                    }
                )
            summary = summarize_run_metrics(run_metrics)
            y, p = aggregate_predictions(frames)
            case_diag = diagnostic_metrics(y, p)
            rows.append(
                {
                    "dataset": dataset,
                    "input": input_name,
                    **summary,
                    "case_avg_auc": case_diag["auc"],
                    "case_avg_bal_acc": case_diag["bal_acc"],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "auto_roi_recomputed.csv", index=False)
    pd.DataFrame(audits).to_csv(out_dir / "auto_roi_label_mismatch_audit.csv", index=False)
    with (out_dir / "auto_roi_rows.tex").open("w", encoding="utf-8") as f:
        for row in rows:
            input_label = {
                "oracle_roi": "Oracle ROI",
                "auto_roi": "Detector ROI",
                "full_image": "Full image",
            }[row["input"]]
            f.write(
                f"{row['dataset']} & {input_label} & "
                f"{row['auc_mean']:.4f} & {row['bal_acc_mean']:.4f} & "
                f"{row['f1_mean']:.4f} & {row['acc_mean']:.4f} \\\\\n"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thyroid-root", type=Path, default=DEFAULT_THYROID_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    audit_rows: list[dict] = []
    pred_frames: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for source in SOURCES:
        add_source_rows(source, args.thyroid_root, labels, run_rows, audit_rows, pred_frames)

    pd.DataFrame(run_rows).to_csv(args.out_dir / "run_level_recomputed_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out_dir / "label_mismatch_audit.csv", index=False)

    records = write_main_tables(args.out_dir, run_rows, pred_frames)
    records["complexity"] = write_complexity_table(args.out_dir, args.thyroid_root, labels)
    records["auto_roi"] = write_auto_roi_table(args.out_dir, labels)
    records["label_snapshot"] = str(args.labels)
    records["output_dir"] = str(args.out_dir)

    with (args.out_dir / "table_values.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote recomputed paper-log-label metrics to {args.out_dir}")
    print(f"Label snapshot: {args.labels}")


if __name__ == "__main__":
    main()
