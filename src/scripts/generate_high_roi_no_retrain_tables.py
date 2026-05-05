#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate high-ROI manuscript tables from frozen prediction CSVs only.

This script is intentionally read-only with respect to experiment data. It does
not load images, current manifests, checkpoints, model weights, or training
entrypoints. The current dataset folders may have drifted after intermediate
updates, so every table here is derived only from completed-result CSV files
whose labels and predictions were frozen at evaluation time.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def detect_project_root() -> Path:
    """Find the original experiment root that contains frozen eval reports."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path.cwd(),
        Path(r"C:\Users\Afr1ste\PycharmProjects\Thyroid"),
    ]
    for candidate in candidates:
        if (candidate / "eval_reports" / "tn5000_auto_roi_final_summary_20260503_161324").exists():
            return candidate
    raise SystemExit(
        "Could not find the original Thyroid experiment root containing frozen eval_reports. "
        "Run this script from the local experiment workspace or use the checked-in generated artifacts."
    )


ROOT = detect_project_root()
OUT_DIR = ROOT / "eval_reports" / "high_roi_no_retrain_20260505"
PAPER_DIR = Path(r"C:\Users\Afr1ste\OneDrive\My Notes\tex\pr_ultrasound_lesion_classification")
PAPER_FIG_DIR = PAPER_DIR / "figures"

MAIN_PREDICTION_SOURCES = {
    "TN5000": [
        ROOT / "tn5000_roi_runs_ggg_mca_enabled_3seed" / "20260426_093735" / "test_predictions.csv",
        ROOT / "tn5000_roi_runs_ggg_mca_enabled_3seed" / "20260426_121126" / "test_predictions.csv",
        ROOT / "tn5000_roi_runs_ggg_mca_enabled_3seed" / "20260426_143912" / "test_predictions.csv",
    ],
    "BUSI": [
        ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_165335" / "test_predictions.csv",
        ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_173346" / "test_predictions.csv",
        ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_181344" / "test_predictions.csv",
        ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_185347" / "test_predictions.csv",
        ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_192506" / "test_predictions.csv",
    ],
    "AUL": [
        ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_200624" / "test_predictions.csv",
        ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_200747" / "test_predictions.csv",
        ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_200911" / "test_predictions.csv",
        ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_201108" / "test_predictions.csv",
        ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_201414" / "test_predictions.csv",
    ],
}


@dataclass(frozen=True)
class Prediction:
    image_id: str
    label: int
    prob: float
    threshold: float


def read_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f4(x: float) -> str:
    return f"{x:.4f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def auc_score(labels: list[int], probs: list[float]) -> float:
    n = len(labels)
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = sorted(range(n), key=lambda i: probs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and probs[order[j]] == probs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_pos_ranks = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)


def confusion_metrics(labels: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / len(labels) if labels else float("nan")
    bal_acc = (sens + spec) / 2.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    f1_neg = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else float("nan")
    f1_macro = (f1_pos + f1_neg) / 2.0
    return {
        "auc": auc_score(labels, probs),
        "bal_acc": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1_macro": f1_macro,
        "acc": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def expected_calibration_error(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    n = len(labels)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            idx = [i for i, p in enumerate(probs) if lo <= p <= hi]
        else:
            idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        obs = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(obs - conf)
    return ece


def brier_score(labels: list[int], probs: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probs)) / len(labels)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def ci95(values: list[float]) -> tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


def load_predictions(path: Path) -> list[Prediction]:
    rows = read_dicts(path)
    preds: list[Prediction] = []
    for row in rows:
        preds.append(
            Prediction(
                image_id=row["image_id"],
                label=int(float(row["true_label"])),
                prob=float(row["prob_class1"]),
                threshold=float(row.get("threshold") or 0.5),
            )
        )
    return preds


def averaged_case_predictions(paths: list[Path]) -> tuple[list[str], list[int], list[float], float]:
    by_id: dict[str, list[Prediction]] = defaultdict(list)
    thresholds: list[float] = []
    for path in paths:
        run_preds = load_predictions(path)
        if not run_preds:
            raise ValueError(f"empty prediction file: {path}")
        thresholds.append(run_preds[0].threshold)
        for pred in run_preds:
            by_id[pred.image_id].append(pred)

    ids = sorted(by_id)
    labels: list[int] = []
    probs: list[float] = []
    for image_id in ids:
        group = by_id[image_id]
        label_set = {p.label for p in group}
        if len(label_set) != 1:
            raise ValueError(f"inconsistent labels for {image_id}: {label_set}")
        if len(group) != len(paths):
            raise ValueError(f"{image_id} has {len(group)} predictions, expected {len(paths)}")
        labels.append(group[0].label)
        probs.append(sum(p.prob for p in group) / len(group))
    return ids, labels, probs, sum(thresholds) / len(thresholds)


def bootstrap_case_metrics(
    labels: list[int],
    probs: list[float],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 20260505,
) -> dict[str, tuple[float, float]]:
    rng = Random(seed)
    n = len(labels)
    buckets: dict[str, list[float]] = defaultdict(list)
    attempts = 0
    while min((len(v) for v in buckets.values()), default=0) < n_boot and attempts < n_boot * 5:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        y = [labels[i] for i in idx]
        if sum(y) == 0 or sum(y) == len(y):
            continue
        p = [probs[i] for i in idx]
        m = confusion_metrics(y, p, threshold)
        m["ece_10"] = expected_calibration_error(y, p, 10)
        m["brier"] = brier_score(y, p)
        for key, val in m.items():
            if isinstance(val, float) and not math.isnan(val):
                buckets[key].append(val)
    return {key: ci95(vals[:n_boot]) for key, vals in buckets.items()}


def reliability_bins(labels: list[int], probs: list[float], n_bins: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            idx = [i for i, p in enumerate(probs) if lo <= p <= hi]
        else:
            idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if idx:
            mean_prob = sum(probs[i] for i in idx) / len(idx)
            frac_pos = sum(labels[i] for i in idx) / len(idx)
        else:
            mean_prob = (lo + hi) / 2.0
            frac_pos = float("nan")
        rows.append(
            {
                "bin": b,
                "bin_low": lo,
                "bin_high": hi,
                "n": len(idx),
                "mean_probability": mean_prob,
                "observed_positive_fraction": frac_pos,
            }
        )
    return rows


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_reliability_figure(
    dataset_bins: dict[str, list[dict[str, object]]],
    dataset_metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    width, height = 1800, 650
    margin_left, margin_top = 120, 125
    panel_w, panel_h = 475, 380
    gap = 65
    bg = "white"
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    title_font = load_font(32, bold=True)
    axis_font = load_font(20)
    tick_font = load_font(17)
    small_font = load_font(18)
    colors = {"TN5000": "#1f77b4", "BUSI": "#2ca02c", "AUL": "#d62728"}

    draw.text((width // 2, 24), "Case-Level Reliability Diagrams", fill="#222222", font=title_font, anchor="ma")

    for panel_idx, dataset in enumerate(["TN5000", "BUSI", "AUL"]):
        x0 = margin_left + panel_idx * (panel_w + gap)
        y0 = margin_top
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), outline="#333333", width=2)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x = x0 + t * panel_w
            y = y1 - t * panel_h
            draw.line((x0, y, x1, y), fill="#dddddd", width=1)
            draw.line((x, y0, x, y1), fill="#eeeeee", width=1)
            draw.text((x, y1 + 10), f"{t:.2g}", fill="#444444", font=tick_font, anchor="ma")
            draw.text((x0 - 10, y), f"{t:.2g}", fill="#444444", font=tick_font, anchor="rm")
        draw.line((x0, y1, x1, y0), fill="#777777", width=2)

        rows = dataset_bins[dataset]
        max_n = max(int(r["n"]) for r in rows) if rows else 1
        for row in rows:
            frac = float(row["observed_positive_fraction"])
            if math.isnan(frac):
                continue
            prob = float(row["mean_probability"])
            n = int(row["n"])
            px = x0 + prob * panel_w
            py = y1 - frac * panel_h
            radius = 7 + int(18 * math.sqrt(n / max_n))
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=colors[dataset],
                outline="white",
                width=3,
            )

        metrics = dataset_metrics[dataset]
        draw.text((x0 + panel_w / 2, y0 - 46), dataset, fill="#222222", font=title_font, anchor="ma")
        metric_box = (
            x0 + 18,
            y0 + 18,
            x0 + 170,
            y0 + 88,
        )
        draw.rounded_rectangle(metric_box, radius=8, fill="#ffffff", outline="#cccccc")
        draw.text(
            (metric_box[0] + 12, metric_box[1] + 10),
            f"ECE={metrics['ece_10']:.3f}\nBrier={metrics['brier']:.3f}",
            fill="#222222",
            font=small_font,
        )
        draw.text((x0 + panel_w / 2, y1 + 54), "Mean predicted probability", fill="#222222", font=axis_font, anchor="ma")
        if panel_idx == 0:
            draw.text(
                (x0, y0 - 18),
                "Observed positive fraction",
                fill="#222222",
                font=small_font,
                anchor="la",
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def generate_case_level_statistics() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    all_bins: dict[str, list[dict[str, object]]] = {}
    all_metrics: dict[str, dict[str, float]] = {}

    for dataset, paths in MAIN_PREDICTION_SOURCES.items():
        ids, labels, probs, threshold = averaged_case_predictions(paths)
        metrics = confusion_metrics(labels, probs, threshold)
        metrics["ece_10"] = expected_calibration_error(labels, probs, 10)
        metrics["brier"] = brier_score(labels, probs)
        cis = bootstrap_case_metrics(labels, probs, threshold)

        row: dict[str, object] = {
            "dataset": dataset,
            "n_cases": len(ids),
            "n_positive": sum(labels),
            "n_negative": len(labels) - sum(labels),
            "n_runs_averaged": len(paths),
            "threshold_mean": threshold,
        }
        for key in [
            "auc",
            "bal_acc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1_macro",
            "acc",
            "ece_10",
            "brier",
        ]:
            row[key] = metrics[key]
            if key in cis:
                row[f"{key}_ci_low"] = cis[key][0]
                row[f"{key}_ci_high"] = cis[key][1]
        summary_rows.append(row)

        bins = reliability_bins(labels, probs, 10)
        for b in bins:
            curve_rows.append({"dataset": dataset, **b})
        all_bins[dataset] = bins
        all_metrics[dataset] = metrics

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_reliability_figure(all_bins, all_metrics, OUT_DIR / "fig_reliability_diagram_case_level.png")
    draw_reliability_figure(all_bins, all_metrics, PAPER_FIG_DIR / "fig_reliability_diagram_case_level.png")
    return summary_rows, curve_rows


def generate_tn5000_oracle_auto_full() -> list[dict[str, object]]:
    metrics_path = ROOT / "eval_reports" / "tn5000_auto_roi_bboxjitter_20260502_165253" / "auto_roi_classification_metrics.csv"
    rows = [r for r in read_dicts(metrics_path) if r["split"] == "test"]
    out: list[dict[str, object]] = []
    name_map = {
        "oracle": "Oracle ROI",
        "auto": "Automatic ROI",
        "full": "Full image",
    }
    for row in rows:
        out.append(
            {
                "input": name_map[row["mode"]],
                "mode": row["mode"],
                "auc": float(row["auc"]),
                "bal_acc": float(row["bal_acc"]),
                "f1_macro": float(row["f1_macro"]),
                "acc": float(row["acc"]),
                "sensitivity": float(row["recall_1"]),
                "specificity": float(row["recall_0"]),
                "threshold": float(row["threshold"]),
                "source": str(metrics_path),
            }
        )
    return out


def generate_tn5000_robustness_rows() -> list[dict[str, object]]:
    path = ROOT / "eval_reports" / "tn5000_auto_roi_final_summary_20260503_161324" / "robustness_probe_summary.csv"
    rows = read_dicts(path)
    out: list[dict[str, object]] = []
    exp_map = {
        "bboxjitter_light": "Box-jitter probe",
        "predboxmix": "Predicted-box mix",
    }
    mode_map = {"oracle": "Oracle ROI", "auto": "Automatic ROI", "full": "Full image"}
    for row in rows:
        if row["mode"] not in mode_map:
            continue
        out.append(
            {
                "probe": exp_map[row["experiment"]],
                "input": mode_map[row["mode"]],
                "auc": float(row["auc"]),
                "bal_acc": float(row["bal_acc"]),
                "f1_macro": float(row["f1_macro"]),
                "acc": float(row["acc"]),
                "threshold": float(row["threshold"]),
                "source": str(path),
            }
        )
    return out


def write_tn5000_oracle_auto_full_tex(path: Path, rows: list[dict[str, object]]) -> None:
    order = ["oracle", "auto", "full"]
    by_mode = {r["mode"]: r for r in rows}
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{TN5000 oracle ROI, automatic ROI, and full-image closed-loop probe from frozen existing predictions. This single-probe table complements the three-seed automatic-ROI summary in Table~\\ref{tab:tn5000_auto_roi_summary}; it uses the same trained classifier family and does not involve retraining.}",
        "    \\label{tab:tn5000_oracle_auto_full_probe}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        Input & AUC & BalAcc & Sens. & Spec. & F1 & Acc \\\\",
        "        \\midrule",
    ]
    for mode in order:
        row = by_mode[mode]
        lines.append(
            f"        {row['input']} & {f4(float(row['auc']))} & {f4(float(row['bal_acc']))} & "
            f"{f4(float(row['sensitivity']))} & {f4(float(row['specificity']))} & "
            f"{f4(float(row['f1_macro']))} & {f4(float(row['acc']))} \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tn5000_robustness_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{TN5000 localization-robustness probes from completed no-retrain evaluations. The probes test whether light box-jitter training or predicted-box mixing improves the oracle/automatic/full-image operating points. They are kept as diagnostic evidence because neither probe clearly improves the automatic-ROI setting enough to replace the original classifier.}",
        "    \\label{tab:tn5000_localization_robustness_probe}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{llcccc}",
        "        \\toprule",
        "        Probe & Input & AUC & BalAcc & F1 & Acc \\\\",
        "        \\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["probe"] != previous:
            lines.append("        \\midrule")
        previous = row["probe"]
        lines.append(
            f"        {row['probe']} & {row['input']} & {f4(float(row['auc']))} & "
            f"{f4(float(row['bal_acc']))} & {f4(float(row['f1_macro']))} & {f4(float(row['acc']))} \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_level_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{Case-level diagnostic statistics for seed/fold-averaged UL-TransXNet predictions on the fixed test sets. Brackets are case-bootstrap 95\\% confidence intervals. Sensitivity, specificity, PPV, and NPV use the mean validation-derived threshold across the contributing seeds or folds.}",
        "    \\label{tab:case_level_diagnostic_statistics}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        Dataset & AUC & Sensitivity & Specificity & PPV & NPV & ECE/Brier \\\\",
        "        \\midrule",
    ]
    for row in rows:
        ece_brier = f"{f3(float(row['ece_10']))}/{f3(float(row['brier']))}"
        lines.append(
            f"        {row['dataset']} & {f4(float(row['auc']))} [{f4(float(row['auc_ci_low']))}, {f4(float(row['auc_ci_high']))}] & "
            f"{f4(float(row['sensitivity']))} [{f4(float(row['sensitivity_ci_low']))}, {f4(float(row['sensitivity_ci_high']))}] & "
            f"{f4(float(row['specificity']))} [{f4(float(row['specificity_ci_low']))}, {f4(float(row['specificity_ci_high']))}] & "
            f"{f4(float(row['ppv']))} [{f4(float(row['ppv_ci_low']))}, {f4(float(row['ppv_ci_high']))}] & "
            f"{f4(float(row['npv']))} [{f4(float(row['npv_ci_low']))}, {f4(float(row['npv_ci_high']))}] & "
            f"{ece_brier} \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    path: Path,
    tn_rows: list[dict[str, object]],
    rob_rows: list[dict[str, object]],
    case_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# High-ROI no-retrain revision outputs",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "All outputs are generated from frozen prediction/result CSV files only. No current dataset manifests, labels, images, checkpoints, or training scripts are loaded.",
        "",
        "## TN5000 oracle/auto/full probe",
        "",
        "| Input | AUC | BalAcc | Sens. | Spec. | F1 | Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tn_rows:
        lines.append(
            f"| {row['input']} | {f4(float(row['auc']))} | {f4(float(row['bal_acc']))} | "
            f"{f4(float(row['sensitivity']))} | {f4(float(row['specificity']))} | "
            f"{f4(float(row['f1_macro']))} | {f4(float(row['acc']))} |"
        )
    lines += [
        "",
        "## TN5000 localization-robustness probes",
        "",
        "| Probe | Input | AUC | BalAcc | F1 | Acc |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rob_rows:
        lines.append(
            f"| {row['probe']} | {row['input']} | {f4(float(row['auc']))} | "
            f"{f4(float(row['bal_acc']))} | {f4(float(row['f1_macro']))} | {f4(float(row['acc']))} |"
        )
    lines += [
        "",
        "## Case-level diagnostic statistics",
        "",
        "| Dataset | n | AUC | Sens. | Spec. | PPV | NPV | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['dataset']} | {row['n_cases']} | {f4(float(row['auc']))} | "
            f"{f4(float(row['sensitivity']))} | {f4(float(row['specificity']))} | "
            f"{f4(float(row['ppv']))} | {f4(float(row['npv']))} | "
            f"{f4(float(row['ece_10']))} | {f4(float(row['brier']))} |"
        )
    lines += [
        "",
        "## Generated files",
        "",
        "- `tn5000_oracle_auto_full_probe.csv`",
        "- `tn5000_oracle_auto_full_probe_table.tex`",
        "- `tn5000_localization_robustness_probe.csv`",
        "- `tn5000_localization_robustness_probe_table.tex`",
        "- `case_level_diagnostic_statistics.csv`",
        "- `case_level_diagnostic_statistics_table.tex`",
        "- `reliability_curve_bins_case_level.csv`",
        "- `fig_reliability_diagram_case_level.png`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tn_rows = generate_tn5000_oracle_auto_full()
    robustness_rows = generate_tn5000_robustness_rows()
    case_rows, curve_rows = generate_case_level_statistics()

    write_dicts(OUT_DIR / "tn5000_oracle_auto_full_probe.csv", tn_rows)
    write_dicts(OUT_DIR / "tn5000_localization_robustness_probe.csv", robustness_rows)
    write_dicts(OUT_DIR / "case_level_diagnostic_statistics.csv", case_rows)
    write_dicts(OUT_DIR / "reliability_curve_bins_case_level.csv", curve_rows)

    write_tn5000_oracle_auto_full_tex(OUT_DIR / "tn5000_oracle_auto_full_probe_table.tex", tn_rows)
    write_tn5000_robustness_tex(OUT_DIR / "tn5000_localization_robustness_probe_table.tex", robustness_rows)
    write_case_level_tex(OUT_DIR / "case_level_diagnostic_statistics_table.tex", case_rows)
    write_summary(OUT_DIR / "README_high_roi_no_retrain.md", tn_rows, robustness_rows, case_rows)

    print(f"Wrote high-ROI no-retrain tables to {OUT_DIR}")
    print(f"Wrote reliability figure to {PAPER_FIG_DIR / 'fig_reliability_diagram_case_level.png'}")


if __name__ == "__main__":
    main()
