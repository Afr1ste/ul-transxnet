#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate manuscript revision tables from existing completed-result CSVs.

This is intentionally read-only with respect to experiment data: it does not
load images, labels, manifests, checkpoints, or model weights, and it never
launches training. It only aggregates frozen result logs that already contain
validation/test metrics.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eval_reports" / "no_retrain_revision_20260505"


def read_rows(path: str) -> list[dict[str, str]]:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: list[dict[str, str]], display_name: str | None = None) -> list[dict[str, str]]:
    if display_name is None:
        return rows
    return [r for r in rows if (r.get("display_name_cfg") or "").strip() == display_name]


def avg(rows: list[dict[str, str]], col: str) -> float:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    if not vals:
        raise ValueError(f"no values for {col}")
    return mean(vals)


def sd(rows: list[dict[str, str]], col: str) -> float:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return stdev(vals) if len(vals) > 1 else 0.0


def fmt(x: float) -> str:
    return f"{x:.4f}"


def fmt_pm(m: float, s: float) -> str:
    return f"{m:.4f} $\\pm$ {s:.4f}"


def tex_pm(value: str) -> str:
    return value.replace("+/-", "$\\pm$")


SOURCES = {
    "TransXNet": {
        "TN5000": ("tn5000_p0_structure_current_3seed_logs/20260429_030247/all_runs_metrics.csv", "TransXNet"),
        "BUSI": ("busi_p0_structure_clean_5fold_logs/20260425_034843/all_runs_metrics.csv", "TransXNet"),
        "AUL": ("aul_p0_structure_clean_5fold_logs/20260425_030033/all_runs_metrics.csv", "TransXNet"),
    },
    "TransXNet-GG": {
        "TN5000": ("tn5000_p0_structure_current_3seed_logs/20260429_104255/all_runs_metrics.csv", "TransXNet-GG"),
        "BUSI": ("busi_p0_structure_clean_5fold_logs/20260425_034843/all_runs_metrics.csv", "TransXNet-GG"),
        "AUL": ("aul_p0_structure_clean_5fold_logs/20260425_030033/all_runs_metrics.csv", "TransXNet-GG"),
    },
    "GGG-noMCA": {
        "TN5000": ("tn5000_ggg_nomca_current_3seed_logs/20260427_051327/all_runs_metrics.csv", None),
        "BUSI": ("busi_ggg_nomca_clean_5fold_safe_logs/20260427_123519/all_runs_metrics.csv", None),
        "AUL": ("aul_ggg_nomca_clean_5fold_safe_logs/20260427_140214/all_runs_metrics.csv", None),
    },
    "UL-TransXNet": {
        "TN5000": ("tn5000_ggg_mca_enabled_3seed_logs/20260426_093728/all_runs_metrics.csv", None),
        "BUSI": ("busi_ggg_mca_clean_5fold_safe_logs/20260426_165332/all_runs_metrics.csv", None),
        "AUL": ("aul_ggg_mca_clean_5fold_safe_logs/20260426_200618/all_runs_metrics.csv", None),
    },
}


def collect_variant_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for variant, by_dataset in SOURCES.items():
        per_dataset: dict[str, dict[str, float | int | str]] = {}
        for dataset, (path, display_name) in by_dataset.items():
            rows = filter_rows(read_rows(path), display_name)
            if not rows:
                raise ValueError(f"no rows for {variant}/{dataset} from {path}")
            per_dataset[dataset] = {
                "n": len(rows),
                "val_auc": avg(rows, "val_auc"),
                "val_bal_acc": avg(rows, "val_bal_acc"),
                "test_auc": avg(rows, "test_auc"),
                "test_bal_acc": avg(rows, "test_bal_acc"),
                "test_auc_sd": sd(rows, "test_auc"),
                "test_bal_acc_sd": sd(rows, "test_bal_acc"),
                "source": path,
            }
        out.append(
            {
                "variant": variant,
                "tn5000_val_auc": per_dataset["TN5000"]["val_auc"],
                "busi_val_auc": per_dataset["BUSI"]["val_auc"],
                "aul_val_auc": per_dataset["AUL"]["val_auc"],
                "mean_val_auc": mean([per_dataset[d]["val_auc"] for d in ["TN5000", "BUSI", "AUL"]]),
                "tn5000_val_bal_acc": per_dataset["TN5000"]["val_bal_acc"],
                "busi_val_bal_acc": per_dataset["BUSI"]["val_bal_acc"],
                "aul_val_bal_acc": per_dataset["AUL"]["val_bal_acc"],
                "mean_val_bal_acc": mean([per_dataset[d]["val_bal_acc"] for d in ["TN5000", "BUSI", "AUL"]]),
                "tn5000_test_auc": per_dataset["TN5000"]["test_auc"],
                "busi_test_auc": per_dataset["BUSI"]["test_auc"],
                "aul_test_auc": per_dataset["AUL"]["test_auc"],
                "mean_test_auc": mean([per_dataset[d]["test_auc"] for d in ["TN5000", "BUSI", "AUL"]]),
                "tn5000_test_bal_acc": per_dataset["TN5000"]["test_bal_acc"],
                "busi_test_bal_acc": per_dataset["BUSI"]["test_bal_acc"],
                "aul_test_bal_acc": per_dataset["AUL"]["test_bal_acc"],
                "mean_test_bal_acc": mean([per_dataset[d]["test_bal_acc"] for d in ["TN5000", "BUSI", "AUL"]]),
                "selected": "yes" if variant == "UL-TransXNet" else "no",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{Validation-performance audit for the TransXNet-family selection. Values are averages over three TN5000 seeds or five BUSI/AUL training folds. The selected row is the final UL-TransXNet configuration used in the main manuscript tables.}",
        "    \\label{tab:validation_selection_audit}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{lccccc}",
        "        \\toprule",
        "        Variant & TN5000 val AUC & BUSI val AUC & AUL val AUC & Mean val AUC & Selected \\\\",
        "        \\midrule",
    ]
    best = max(float(r["mean_val_auc"]) for r in rows)
    for r in rows:
        bold = abs(float(r["mean_val_auc"]) - best) < 1e-12
        mean_val = fmt(float(r["mean_val_auc"]))
        if bold:
            mean_val = f"\\textbf{{{mean_val}}}"
        selected = "yes" if r["selected"] == "yes" else "no"
        lines.append(
            f"        {r['variant']} & {fmt(float(r['tn5000_val_auc']))} & {fmt(float(r['busi_val_auc']))} & "
            f"{fmt(float(r['aul_val_auc']))} & {mean_val} & {selected} \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_transxnet_delta_tex(path: Path) -> None:
    rows = collect_variant_rows()
    base = next(r for r in rows if r["variant"] == "TransXNet")
    ours = next(r for r in rows if r["variant"] == "UL-TransXNet")
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{Original TransXNet versus UL-TransXNet under the same dataset-specific reporting protocol. $\\Delta$ denotes UL-TransXNet minus TransXNet.}",
        "    \\label{tab:original_transxnet_delta}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        Dataset & \\multicolumn{2}{c}{TransXNet} & \\multicolumn{2}{c}{UL-TransXNet} & \\multicolumn{2}{c}{$\\Delta$} \\\\",
        "        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "        & AUC & BalAcc & AUC & BalAcc & AUC & BalAcc \\\\",
        "        \\midrule",
    ]
    for ds, key in [("TN5000", "tn5000"), ("BUSI", "busi"), ("AUL", "aul")]:
        base_auc = float(base[f"{key}_test_auc"])
        base_bal = float(base[f"{key}_test_bal_acc"])
        ours_auc = float(ours[f"{key}_test_auc"])
        ours_bal = float(ours[f"{key}_test_bal_acc"])
        lines.append(
            f"        {ds} & {fmt(base_auc)} & {fmt(base_bal)} & {fmt(ours_auc)} & {fmt(ours_bal)} & "
            f"{fmt(ours_auc - base_auc)} & {fmt(ours_bal - base_bal)} \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_tn5000_auto_roi_tex(path: Path) -> None:
    loc_rows = list(csv.DictReader((ROOT / "eval_reports/tn5000_auto_roi_final_summary_20260503_161324/detector_localization_summary.csv").open("r", encoding="utf-8-sig")))
    cls_rows = list(csv.DictReader((ROOT / "eval_reports/tn5000_auto_roi_final_summary_20260503_161324/closed_loop_classification_summary.csv").open("r", encoding="utf-8-sig")))
    loc = {r["metric"]: r["mean_std"] for r in loc_rows}
    cls = {r["metric"]: r["mean_std"] for r in cls_rows}
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{TN5000 automatic ROI sanity check using existing detector and classifier outputs. Detector values summarize three YOLO11n detector seeds; classification values summarize automatic-ROI closed-loop evaluation across the three detector seeds. No classifier retraining is involved.}",
        "    \\label{tab:tn5000_auto_roi_summary}",
        "    \\begin{adjustbox}{max width=\\columnwidth}",
        "    \\begin{tabular}{lcc}",
        "        \\toprule",
        "        Block & Metric & Value \\\\",
        "        \\midrule",
        f"        Detector & mAP50 & {tex_pm(loc['test_map50'])} \\\\",
        f"        Detector & mean IoU & {tex_pm(loc['test_mean_iou'])} \\\\",
        f"        Detector & R@0.75 & {tex_pm(loc['test_recall_iou_0_75'])} \\\\",
        f"        Detector & no-detection rate & {tex_pm(loc['test_no_detection_rate'])} \\\\",
        "        \\midrule",
        f"        Auto ROI classifier & AUC & {tex_pm(cls['test_auc'])} \\\\",
        f"        Auto ROI classifier & BalAcc & {tex_pm(cls['test_bal_acc'])} \\\\",
        f"        Auto ROI classifier & F1-macro & {tex_pm(cls['test_f1_macro'])} \\\\",
        f"        Auto ROI classifier & Acc & {tex_pm(cls['test_acc'])} \\\\",
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# No-retrain revision tables",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "These files were generated from existing completed-result CSVs only. No images, manifests, checkpoints, or training entrypoints were loaded. This is intentional because the current intermediate dataset state may contain label drift relative to the frozen result logs.",
        "",
        "## Key validation-selection audit",
        "",
        "| Variant | TN5000 val AUC | BUSI val AUC | AUL val AUC | Mean val AUC | Selected |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {fmt(float(r['tn5000_val_auc']))} | {fmt(float(r['busi_val_auc']))} | "
            f"{fmt(float(r['aul_val_auc']))} | {fmt(float(r['mean_val_auc']))} | {r['selected']} |"
        )
    lines += [
        "",
        "## Sources",
        "",
    ]
    for variant, by_dataset in SOURCES.items():
        for dataset, (source, display_name) in by_dataset.items():
            filt = f" display_name_cfg={display_name}" if display_name else ""
            lines.append(f"- {variant} / {dataset}: `{source}`{filt}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_variant_rows()
    write_csv(OUT_DIR / "validation_selection_audit.csv", rows)
    write_selection_tex(OUT_DIR / "validation_selection_audit_table.tex", rows)
    write_transxnet_delta_tex(OUT_DIR / "original_transxnet_delta_table.tex")
    copy_tn5000_auto_roi_tex(OUT_DIR / "tn5000_auto_roi_summary_table.tex")
    write_markdown_summary(OUT_DIR / "README_no_retrain_revision.md", rows)
    print(f"Wrote no-retrain revision tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
