import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, brier_score_loss

ROOT = Path(r"C:\Users\Afr1ste\PycharmProjects\Thyroid")
MAIN_RUN_METRICS = {
    "TN5000": ROOT / r"tn5000_ggg_mca_enabled_3seed_logs\20260426_093728\all_runs_metrics.csv",
    "BUSI": ROOT / r"busi_ggg_mca_clean_5fold_safe_logs\20260426_165332\all_runs_metrics.csv",
    "AUL": ROOT / r"aul_ggg_mca_clean_5fold_safe_logs\20260426_200618\all_runs_metrics.csv",
}
DEFAULT_CLOSED_LOOP_ROOT = ROOT / r"eval_reports\busi_aul_closed_loop_auto_roi_20260504_165336"


def read_pred_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_id": r.get("image_id", ""),
                "true_label": int(float(r["true_label"])),
                "pred_label": int(float(r["pred_label"])),
                "prob_class1": float(r["prob_class1"]),
                "threshold": float(r.get("threshold", 0.5) or 0.5),
            })
    return rows


def ece_score(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def metrics_for_rows(rows, threshold=None):
    y = np.array([r["true_label"] for r in rows], dtype=int)
    p = np.array([r["prob_class1"] for r in rows], dtype=float)
    if threshold is None:
        pred = np.array([r["pred_label"] for r in rows], dtype=int)
    else:
        pred = (p >= threshold).astype(int)
    out = {}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    out["bal_acc"] = float(balanced_accuracy_score(y, pred))
    out["f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
    out["acc"] = float(accuracy_score(y, pred))
    out["brier"] = float(brier_score_loss(y, p))
    out["ece_10"] = ece_score(y, p, n_bins=10)
    return out


def bootstrap_ci_values(vals, n_boot=10000, seed=20260504):
    rng = random.Random(seed)
    vals = list(map(float, vals))
    n = len(vals)
    boots = [mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def read_main_run_prediction_paths(metrics_csv):
    paths = []
    with open(metrics_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") and r["status"] != "ok":
                continue
            pred_path = Path(r["run_dir"]) / "test_predictions.csv"
            if pred_path.exists():
                paths.append(pred_path)
    return paths


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.4f}"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "eval_reports" / "paper_statistical_calibration_20260504"))
    ap.add_argument("--closed-loop-root", default=str(DEFAULT_CLOSED_LOOP_ROOT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    closed_loop_root = Path(args.closed_loop_root)

    main_rows = []
    for dataset, metrics_csv in MAIN_RUN_METRICS.items():
        pred_paths = read_main_run_prediction_paths(metrics_csv)
        if not pred_paths:
            raise RuntimeError(f"No run-level predictions found for {dataset}: {metrics_csv}")
        run_recs = []
        for path in pred_paths:
            rows = read_pred_csv(path)
            run_recs.append({"path": str(path), "n": len(rows), **metrics_for_rows(rows)})
        row = {
            "dataset": dataset,
            "n_runs": len(run_recs),
            "n_per_run": ";".join(str(r["n"]) for r in run_recs),
            "source": str(metrics_csv),
            "prediction_sources": "|".join(r["path"] for r in run_recs),
        }
        for metric in ["auc", "bal_acc", "f1_macro", "acc", "ece_10", "brier"]:
            vals = [r[metric] for r in run_recs]
            lo, hi = bootstrap_ci_values(vals, seed=20260504)
            row[metric] = mean(vals)
            row[f"{metric}_std"] = pstdev(vals)
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        main_rows.append(row)

    fold_metrics = {}
    for dataset in ["busi", "aul"]:
        for mode in ["oracle", "auto", "full"]:
            recs = []
            for fold in range(5):
                path = closed_loop_root / dataset / mode / f"fold{fold}" / "test_predictions.csv"
                rows = read_pred_csv(path)
                recs.append({"dataset": dataset.upper(), "mode": mode, "fold": fold, "n": len(rows), **metrics_for_rows(rows), "source": str(path)})
            fold_metrics[(dataset, mode)] = recs

    closed_rows = []
    for dataset in ["busi", "aul"]:
        for mode in ["oracle", "auto", "full"]:
            recs = fold_metrics[(dataset, mode)]
            for metric in ["auc", "bal_acc", "f1_macro", "acc", "ece_10", "brier"]:
                vals = [r[metric] for r in recs]
                lo, hi = bootstrap_ci_values(vals, seed=20260504)
                closed_rows.append({
                    "dataset": dataset.upper(), "mode": mode, "metric": metric,
                    "n_folds": 5, "mean": mean(vals), "std": pstdev(vals),
                    "ci_low": lo, "ci_high": hi,
                })

    delta_rows = []
    for dataset in ["busi", "aul"]:
        for lhs, rhs in [("auto", "full"), ("auto", "oracle")]:
            for metric in ["auc", "bal_acc", "f1_macro", "acc", "ece_10", "brier"]:
                vals = [fold_metrics[(dataset, lhs)][i][metric] - fold_metrics[(dataset, rhs)][i][metric] for i in range(5)]
                lo, hi = bootstrap_ci_values(vals, seed=20260506)
                delta_rows.append({
                    "dataset": dataset.upper(), "comparison": f"{lhs}-{rhs}", "metric": metric,
                    "n_folds": 5, "mean_delta": mean(vals), "std_delta": pstdev(vals),
                    "ci_low": lo, "ci_high": hi,
                })

    write_csv(out_dir / "main_model_bootstrap_calibration.csv", main_rows, list(main_rows[0].keys()))
    write_csv(out_dir / "auto_roi_closed_loop_metric_ci.csv", closed_rows, list(closed_rows[0].keys()))
    write_csv(out_dir / "auto_roi_closed_loop_delta_ci.csv", delta_rows, list(delta_rows[0].keys()))

    auto = {(r["dataset"], r["mode"], r["metric"]): r for r in closed_rows}
    delta = {(r["dataset"], r["comparison"], r["metric"]): r for r in delta_rows}
    row_end = r" \\",
    row_end = " \\\\"  # two LaTeX backslashes

    auto_tex = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \small",
        r"    \caption{Closed-loop classification with oracle ROI, automatic detector ROI, and full-image input. Values are mean $\pm$ standard deviation across five folds. Bracketed intervals are fold-bootstrap 95\% confidence intervals for AUC and balanced accuracy.}",
        r"    \label{tab:auto_roi_closed_loop}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{llcccc}",
        r"        \toprule",
        r"        Dataset & Input & AUC & BalAcc & F1-macro & Acc \\",
        r"        \midrule",
    ]
    for ds in ["BUSI", "AUL"]:
        for mode in ["oracle", "auto", "full"]:
            auc, bal = auto[(ds, mode, "auc")], auto[(ds, mode, "bal_acc")]
            f1, acc = auto[(ds, mode, "f1_macro")], auto[(ds, mode, "acc")]
            auto_tex.append(
                f"        {ds} & {mode} & {fmt(auc['mean'])} $\\pm$ {fmt(auc['std'])} [{fmt(auc['ci_low'])}, {fmt(auc['ci_high'])}] & "
                f"{fmt(bal['mean'])} $\\pm$ {fmt(bal['std'])} [{fmt(bal['ci_low'])}, {fmt(bal['ci_high'])}] & "
                f"{fmt(f1['mean'])} $\\pm$ {fmt(f1['std'])} & {fmt(acc['mean'])} $\\pm$ {fmt(acc['std'])}{row_end}"
            )
        if ds == "BUSI":
            auto_tex.append(r"        \midrule")
    auto_tex += [r"        \bottomrule", r"    \end{tabular}%", r"    }", r"\end{table*}"]
    (out_dir / "auto_roi_closed_loop_table.tex").write_text("\n".join(auto_tex) + "\n", encoding="utf-8")

    det_tex = r"""\begin{table}[t]
    \centering
    \small
    \caption{Detector localization quality used for automatic ROI construction. Values are mean $\pm$ standard deviation across five detector folds.}
    \label{tab:auto_roi_detector_quality}
    \begin{adjustbox}{max width=\columnwidth}
    \begin{tabular}{lcccc}
        \toprule
        Dataset & Mean IoU & R@0.50 & R@0.75 & No det. \\
        \midrule
        BUSI & 0.7365 $\pm$ 0.0241 & 0.8403 $\pm$ 0.0141 & 0.6822 $\pm$ 0.0548 & 0.0000 \\
        AUL & 0.5755 $\pm$ 0.0221 & 0.6740 $\pm$ 0.0419 & 0.4331 $\pm$ 0.0201 & 0.0000 \\
        \bottomrule
    \end{tabular}
    \end{adjustbox}
\end{table}
"""
    (out_dir / "auto_roi_detector_quality_table.tex").write_text(det_tex, encoding="utf-8")

    calib_tex = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \caption{Uncertainty and calibration summary for UL-TransXNet on the main test sets. Values follow the manuscript seed/fold-level reporting protocol; intervals are bootstrap 95\% confidence intervals over seeds or folds. ECE uses 10 confidence bins.}",
        r"    \label{tab:calibration_uncertainty}",
        r"    \begin{adjustbox}{max width=\columnwidth}",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        Dataset & AUC 95\% CI & BalAcc 95\% CI & ECE & Brier \\",
        r"        \midrule",
    ]
    for r in main_rows:
        calib_tex.append(
            f"        {r['dataset']} & {fmt(r['auc'])} [{fmt(r['auc_ci_low'])}, {fmt(r['auc_ci_high'])}] & "
            f"{fmt(r['bal_acc'])} [{fmt(r['bal_acc_ci_low'])}, {fmt(r['bal_acc_ci_high'])}] & "
            f"{fmt(r['ece_10'])} & {fmt(r['brier'])}{row_end}"
        )
    calib_tex += [r"        \bottomrule", r"    \end{tabular}", r"    \end{adjustbox}", r"\end{table}"]
    (out_dir / "calibration_uncertainty_table.tex").write_text("\n".join(calib_tex) + "\n", encoding="utf-8")

    md = ["# Statistical and calibration supplement", "", f"Output directory: `{out_dir}`", "", "## Main UL-TransXNet uncertainty/calibration"]
    for r in main_rows:
        md.append(f"- {r['dataset']}: AUC {fmt(r['auc'])} [{fmt(r['auc_ci_low'])}, {fmt(r['auc_ci_high'])}], BalAcc {fmt(r['bal_acc'])} [{fmt(r['bal_acc_ci_low'])}, {fmt(r['bal_acc_ci_high'])}], ECE {fmt(r['ece_10'])}, Brier {fmt(r['brier'])}")
    md += ["", "## Automatic ROI key paired deltas"]
    for ds in ["BUSI", "AUL"]:
        for comp in ["auto-full", "auto-oracle"]:
            auc, bal = delta[(ds, comp, "auc")], delta[(ds, comp, "bal_acc")]
            md.append(f"- {ds} {comp}: delta AUC {fmt(auc['mean_delta'])} [{fmt(auc['ci_low'])}, {fmt(auc['ci_high'])}], delta BalAcc {fmt(bal['mean_delta'])} [{fmt(bal['ci_low'])}, {fmt(bal['ci_high'])}]")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
