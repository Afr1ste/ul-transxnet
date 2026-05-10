import csv
import itertools
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"<LOCAL_THYROID_ROOT>")
PAPER_ROOT = Path(r"<LOCAL_MANUSCRIPT_ROOT>")

COMPARISONS = [
    {
        "dataset": "TN5000",
        "key": "seed",
        "ours_name": "UL-TransXNet",
        "base_name": "ConvNeXt-Tiny",
        "ours_csv": ROOT / r"tn5000_ggg_mca_enabled_3seed_logs\20260426_093728\all_runs_metrics.csv",
        "base_csv": ROOT / r"tn5000_compare_extra4models_3seed_logs\20260421_222342_merged_complete\all_runs_metrics.csv",
        "ours_filter": ("display_name_cfg", "Ours-GGG-MCAON"),
        "base_filter": ("display_name_cfg", "ConvNeXt-Tiny"),
    },
    {
        "dataset": "BUSI",
        "key": "fold_idx",
        "ours_name": "UL-TransXNet",
        "base_name": "Swin-T",
        "ours_csv": ROOT / r"busi_ggg_mca_clean_5fold_safe_logs\20260426_165332\all_runs_metrics.csv",
        "base_csv": ROOT / r"busi_compare_5models_5fold_logs\20260403_083238\all_runs_metrics.csv",
        "ours_filter": ("display_name_cfg", "TransXNet-GGG"),
        "base_filter": ("display_name_cfg", "Swin-T"),
    },
    {
        "dataset": "AUL",
        "key": "fold_idx",
        "ours_name": "UL-TransXNet",
        "base_name": "Swin-T",
        "ours_csv": ROOT / r"aul_ggg_mca_clean_5fold_safe_logs\20260426_200618\all_runs_metrics.csv",
        "base_csv": ROOT / r"aul_roi_compare_5models_5fold_logs\20260404_235703\all_runs_metrics.csv",
        "ours_filter": ("display_name_cfg", "TransXNet-GGG"),
        "base_filter": ("display_name_cfg", "Swin-T"),
    },
]

METRICS = [
    ("test_auc", "AUC"),
    ("test_bal_acc", "BalAcc"),
]


def load_filtered(path: Path, flt):
    df = pd.read_csv(path)
    col, val = flt
    if col in df.columns:
        df = df[df[col].astype(str) == str(val)].copy()
    if "status" in df.columns:
        df = df[(df["status"].isna()) | (df["status"].astype(str).str.lower().isin(["", "ok"]))]
    return df


def paired_exact_signflip_p(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    obs = abs(float(np.mean(diffs)))
    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        total += 1
        stat = abs(float(np.mean(diffs * np.asarray(signs))))
        if stat >= obs - 1e-15:
            count += 1
    return count / total


def paired_bootstrap_ci(diffs, n_boot=50000, seed=20260504):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt4(x):
    return f"{float(x):.4f}"


def latex_row(row):
    delta = float(row["delta_mean"])
    sign = "+" if delta > 0 else ""
    return (
        f"        {row['dataset']} & {row['baseline']} & {row['metric']} & {row['n_pairs']} & "
        f"{sign}{delta:.4f} & [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] & {float(row['p_signflip']):.4f} \\\\"  # noqa
    )


def main():
    out_dir = ROOT / "eval_reports" / f"paper_offline_stat_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pair_rows = []
    for cfg in COMPARISONS:
        ours = load_filtered(cfg["ours_csv"], cfg["ours_filter"])
        base = load_filtered(cfg["base_csv"], cfg["base_filter"])
        key = cfg["key"]
        ours[key] = ours[key].astype(int)
        base[key] = base[key].astype(int)
        merged = ours.merge(base, on=key, suffixes=("_ours", "_base"), how="inner")
        if len(merged) == 0:
            raise RuntimeError(f"No paired rows for {cfg['dataset']} by {key}")
        merged = merged.sort_values(key)
        for metric_col, metric_name in METRICS:
            ov = merged[f"{metric_col}_ours"].astype(float).to_numpy()
            bv = merged[f"{metric_col}_base"].astype(float).to_numpy()
            diffs = ov - bv
            ci_low, ci_high = paired_bootstrap_ci(diffs)
            p = paired_exact_signflip_p(diffs)
            rows.append({
                "dataset": cfg["dataset"],
                "baseline": cfg["base_name"],
                "metric": metric_name,
                "n_pairs": len(diffs),
                "ours_mean": float(np.mean(ov)),
                "baseline_mean": float(np.mean(bv)),
                "delta_mean": float(np.mean(diffs)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_signflip": p,
                "pair_key": key,
                "ours_csv": str(cfg["ours_csv"]),
                "baseline_csv": str(cfg["base_csv"]),
            })
        for _, r in merged.iterrows():
            pair_rows.append({
                "dataset": cfg["dataset"],
                "pair_key": key,
                "pair_value": int(r[key]),
                "baseline": cfg["base_name"],
                "ours_auc": float(r["test_auc_ours"]),
                "baseline_auc": float(r["test_auc_base"]),
                "delta_auc": float(r["test_auc_ours"] - r["test_auc_base"]),
                "ours_bal_acc": float(r["test_bal_acc_ours"]),
                "baseline_bal_acc": float(r["test_bal_acc_base"]),
                "delta_bal_acc": float(r["test_bal_acc_ours"] - r["test_bal_acc_base"]),
            })

    rows_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    rows_df.to_csv(out_dir / "paired_signflip_bootstrap_summary.csv", index=False, encoding="utf-8")
    pair_df.to_csv(out_dir / "paired_values.csv", index=False, encoding="utf-8")

    tex_lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \caption{Offline paired statistical comparison between UL-TransXNet and the strongest non-\textit{Ours} baseline in Table~\ref{tab:main_results}. $\Delta$ denotes UL-TransXNet minus baseline. Confidence intervals are paired bootstrap 95\% intervals over seeds or folds; $p$ is an exact two-sided paired sign-flip test. Because only three TN5000 seeds and five BUSI/AUL folds are available, these tests are intended as repeated-run stability checks rather than large-sample clinical significance tests.}",
        r"    \label{tab:paired_statistical_tests}",
        r"    \begin{adjustbox}{max width=\columnwidth}",
        r"    \begin{tabular}{lllcccc}",
        r"        \toprule",
        r"        Dataset & Baseline & Metric & $n$ & $\Delta$ & 95\% CI & $p$ \\",
        r"        \midrule",
    ]
    last_ds = None
    for row in rows:
        if last_ds is not None and row["dataset"] != last_ds:
            tex_lines.append(r"        \midrule")
        tex_lines.append(latex_row(row))
        last_ds = row["dataset"]
    tex_lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \end{adjustbox}",
        r"\end{table}",
    ]
    (out_dir / "paired_statistical_tests_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    md = [
        "# Offline paired statistical tests",
        "",
        "Comparison: UL-TransXNet minus strongest non-Ours baseline from the main result table.",
        "",
        rows_df[["dataset", "baseline", "metric", "n_pairs", "ours_mean", "baseline_mean", "delta_mean", "ci_low", "ci_high", "p_signflip"]].to_markdown(index=False),
        "",
        "## Sources",
    ]
    for cfg in COMPARISONS:
        md.append(f"- {cfg['dataset']} ours: `{cfg['ours_csv']}`")
        md.append(f"- {cfg['dataset']} baseline: `{cfg['base_csv']}`")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    latest = ROOT / "eval_reports" / "paper_offline_stat_tests_latest.txt"
    latest.write_text(str(out_dir), encoding="utf-8")
    print(out_dir)
    print(rows_df[["dataset", "baseline", "metric", "n_pairs", "delta_mean", "ci_low", "ci_high", "p_signflip"]].to_string(index=False))


if __name__ == "__main__":
    main()
