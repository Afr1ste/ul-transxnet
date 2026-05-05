from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_PTR = PROJECT_ROOT / "eval_reports" / "main_result_statistics_latest.txt"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    log_dir: Path


DATASETS = [
    DatasetSpec(
        name="TN5000",
        log_dir=PROJECT_ROOT / "tn5000_compare_5models_3seed_logs" / "20260402_192605",
    ),
    DatasetSpec(
        name="BUSI",
        log_dir=PROJECT_ROOT / "busi_compare_5models_5fold_logs" / "20260403_083238",
    ),
    DatasetSpec(
        name="AUL",
        log_dir=PROJECT_ROOT / "aul_roi_compare_5models_5fold_logs" / "20260404_235703",
    ),
]

DISPLAY_MAP = {
    "OURS": "Ours",
    "RESNET50": "ResNet50",
    "EFFB0": "EfficientNet-B0",
    "MBV3": "MobileNetV3-Large",
    "SWINT": "Swin-T",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build bootstrap CI and paired significance tables for TN5000/BUSI/AUL main results."
    )
    p.add_argument("--num-bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260415)
    p.add_argument("--skip-if-complete", type=int, default=1)
    p.add_argument("--output-dir", default="")
    p.add_argument("--tn5000-log-dir", default=str(DATASETS[0].log_dir))
    p.add_argument("--busi-log-dir", default=str(DATASETS[1].log_dir))
    p.add_argument("--aul-log-dir", default=str(DATASETS[2].log_dir))
    return p.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    return [
        DatasetSpec("TN5000", Path(args.tn5000_log_dir)),
        DatasetSpec("BUSI", Path(args.busi_log_dir)),
        DatasetSpec("AUL", Path(args.aul_log_dir)),
    ]


def latest_complete_output(required_bootstrap: int) -> Path | None:
    if not LATEST_PTR.exists():
        return None
    out_dir = Path(LATEST_PTR.read_text(encoding="utf-8").strip())
    required = [
        out_dir / "paper_main_result_ci_table.csv",
        out_dir / "paper_ours_vs_best_baseline_stats.csv",
        out_dir / "README_main_result_statistics.md",
        out_dir / "summary_manifest.json",
    ]
    if out_dir.exists() and all(p.exists() for p in required):
        manifest = json.loads((out_dir / "summary_manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("num_bootstrap", -1)) == int(required_bootstrap):
            return out_dir
    return None


def write_latest_ptr(out_dir: Path) -> None:
    LATEST_PTR.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PTR.write_text(str(out_dir), encoding="utf-8")


def detect_display_name(model_dir_name: str) -> str:
    upper = model_dir_name.upper()
    for prefix, display in DISPLAY_MAP.items():
        if upper.startswith(prefix):
            return display
    return model_dir_name


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise RuntimeError(f"Empty predictions csv: {csv_path}")
    y_true = np.array([int(r["true_label"]) for r in rows], dtype=np.int64)
    prob1 = np.array([float(r["prob_class1"]) for r in rows], dtype=np.float64)
    threshold = float(rows[0]["threshold"])
    return y_true, prob1, threshold


def compute_metrics(y_true: np.ndarray, prob1: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob1 >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auc": float(roc_auc_score(y_true, prob1)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, average="macro")),
        "acc": float(accuracy_score(y_true, pred)),
        "sens": float(sens),
        "spec": float(spec),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_neg": int((y_true == 0).sum()),
        "n_pos": int((y_true == 1).sum()),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx0 = np.flatnonzero(y_true == 0)
    idx1 = np.flatnonzero(y_true == 1)
    sample0 = rng.choice(idx0, size=len(idx0), replace=True)
    sample1 = rng.choice(idx1, size=len(idx1), replace=True)
    out = np.concatenate([sample0, sample1])
    rng.shuffle(out)
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    prob1: np.ndarray,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    metrics = {"auc": [], "bal_acc": [], "f1": [], "acc": [], "sens": [], "spec": []}
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y_true, rng)
        sample = compute_metrics(y_true[idx], prob1[idx], threshold)
        for key in metrics:
            metrics[key].append(sample[key])
    ci = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=np.float64)
        ci[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def paired_bootstrap_stats(
    y_true: np.ndarray,
    ours_prob1: np.ndarray,
    ours_threshold: float,
    base_prob1: np.ndarray,
    base_threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    diffs = {key: [] for key in ("auc", "bal_acc", "f1", "acc", "sens", "spec")}
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y_true, rng)
        ours = compute_metrics(y_true[idx], ours_prob1[idx], ours_threshold)
        base = compute_metrics(y_true[idx], base_prob1[idx], base_threshold)
        for key in diffs:
            diffs[key].append(ours[key] - base[key])

    out: dict[str, float] = {}
    for key, values in diffs.items():
        arr = np.array(values, dtype=np.float64)
        lower = float(np.percentile(arr, 2.5))
        upper = float(np.percentile(arr, 97.5))
        p_two_sided = float(2.0 * min((arr <= 0).mean(), (arr >= 0).mean()))
        out[f"delta_{key}"] = float(arr.mean())
        out[f"delta_{key}_ci_low"] = lower
        out[f"delta_{key}_ci_high"] = upper
        out[f"p_{key}"] = min(p_two_sided, 1.0)
    return out


def fmt_ci(value: float, ci_low: float, ci_high: float) -> str:
    return f"{value:.4f} [{ci_low:.4f}, {ci_high:.4f}]"


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_readme(
    out_dir: Path,
    ci_rows: list[dict],
    compare_rows: list[dict],
    n_boot: int,
    datasets: list[DatasetSpec],
) -> None:
    lines = []
    lines.append("# Main Result Statistics")
    lines.append("")
    lines.append(f"- Bootstraps: {n_boot}")
    lines.append(f"- Output dir: {out_dir}")
    lines.append("")
    lines.append("## Dataset Sources")
    for ds in datasets:
        lines.append(f"- {ds.name}: {ds.log_dir}")
    lines.append("")
    lines.append("## CI Summary")
    for row in ci_rows:
        lines.append(
            f"- {row['dataset']} | {row['model']}: "
            f"AUC {row['auc_ci_str']}, "
            f"BalAcc {row['bal_acc_ci_str']}, "
            f"F1 {row['f1_ci_str']}"
        )
    lines.append("")
    lines.append("## Ours vs Best Non-Ours")
    for row in compare_rows:
        lines.append(
            f"- {row['dataset']}: baseline={row['baseline_model']}, "
            f"DeltaAUC={row['delta_auc']:.4f} (p={row['p_auc']:.4g}), "
            f"DeltaBalAcc={row['delta_bal_acc']:.4f} (p={row['p_bal_acc']:.4g}), "
            f"DeltaF1={row['delta_f1']:.4f} (p={row['p_f1']:.4g})"
        )
    (out_dir / "README_main_result_statistics.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args)

    if bool(args.skip_if_complete):
        prev = latest_complete_output(int(args.num_bootstrap))
        if prev is not None and not str(args.output_dir).strip():
            print(f"[SKIP] Existing complete main-result statistics found: {prev}")
            return

    if str(args.output_dir).strip():
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "eval_reports" / f"main_result_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ci_rows: list[dict] = []
    compare_rows: list[dict] = []
    manifest: dict[str, dict] = {"datasets": {}, "num_bootstrap": int(args.num_bootstrap)}
    rng_master = np.random.default_rng(args.seed)

    for ds in datasets:
        model_level_root = ds.log_dir / "model_level"
        if not model_level_root.exists():
            raise FileNotFoundError(f"Missing model_level directory: {model_level_root}")

        model_items = []
        for model_dir in sorted([p for p in model_level_root.iterdir() if p.is_dir()]):
            pred_csv = model_dir / "test_ensemble_predictions.csv"
            if not pred_csv.exists():
                continue
            display_name = detect_display_name(model_dir.name)
            y_true, prob1, threshold = load_predictions(pred_csv)
            metrics = compute_metrics(y_true, prob1, threshold)
            ci = bootstrap_ci(
                y_true=y_true,
                prob1=prob1,
                threshold=threshold,
                n_boot=int(args.num_bootstrap),
                rng=np.random.default_rng(rng_master.integers(1, 2**32 - 1)),
            )
            row = {
                "dataset": ds.name,
                "model": display_name,
                "model_dir": str(model_dir),
                "prediction_csv": str(pred_csv),
                "n": metrics["n"],
                "n_neg": metrics["n_neg"],
                "n_pos": metrics["n_pos"],
                "threshold": metrics["threshold"],
            }
            for key in ("auc", "bal_acc", "f1", "acc", "sens", "spec"):
                row[key] = metrics[key]
                row[f"{key}_ci_low"] = ci[key][0]
                row[f"{key}_ci_high"] = ci[key][1]
                row[f"{key}_ci_str"] = fmt_ci(metrics[key], ci[key][0], ci[key][1])
            ci_rows.append(row)
            model_items.append(
                {
                    "display_name": display_name,
                    "y_true": y_true,
                    "prob1": prob1,
                    "threshold": threshold,
                    "metrics": metrics,
                }
            )

        if not model_items:
            raise RuntimeError(f"No model-level predictions found for {ds.name} in {model_level_root}")

        ours = next((x for x in model_items if x["display_name"] == "Ours"), None)
        if ours is None:
            raise RuntimeError(f"Ours predictions missing for {ds.name}")
        non_ours = [x for x in model_items if x["display_name"] != "Ours"]
        best_baseline = max(non_ours, key=lambda x: x["metrics"]["auc"])

        paired = paired_bootstrap_stats(
            y_true=ours["y_true"],
            ours_prob1=ours["prob1"],
            ours_threshold=ours["threshold"],
            base_prob1=best_baseline["prob1"],
            base_threshold=best_baseline["threshold"],
            n_boot=int(args.num_bootstrap),
            rng=np.random.default_rng(rng_master.integers(1, 2**32 - 1)),
        )
        compare_row = {
            "dataset": ds.name,
            "ours_model": "Ours",
            "baseline_model": best_baseline["display_name"],
            "ours_auc": ours["metrics"]["auc"],
            "baseline_auc": best_baseline["metrics"]["auc"],
            "ours_bal_acc": ours["metrics"]["bal_acc"],
            "baseline_bal_acc": best_baseline["metrics"]["bal_acc"],
            "ours_f1": ours["metrics"]["f1"],
            "baseline_f1": best_baseline["metrics"]["f1"],
            "ours_acc": ours["metrics"]["acc"],
            "baseline_acc": best_baseline["metrics"]["acc"],
            "ours_sens": ours["metrics"]["sens"],
            "baseline_sens": best_baseline["metrics"]["sens"],
            "ours_spec": ours["metrics"]["spec"],
            "baseline_spec": best_baseline["metrics"]["spec"],
        }
        compare_row.update(paired)
        compare_rows.append(compare_row)

        manifest["datasets"][ds.name] = {
            "log_dir": str(ds.log_dir),
            "best_non_ours_by_auc": best_baseline["display_name"],
        }

    ci_rows.sort(key=lambda x: (x["dataset"], -x["auc"], -x["bal_acc"]))
    compare_rows.sort(key=lambda x: x["dataset"])

    save_csv(out_dir / "paper_main_result_ci_table.csv", ci_rows)
    save_csv(out_dir / "paper_ours_vs_best_baseline_stats.csv", compare_rows)
    build_readme(out_dir, ci_rows, compare_rows, int(args.num_bootstrap), datasets)
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_latest_ptr(out_dir)

    print("=" * 100)
    print("[DONE] main result statistics built")
    print(f"[OUT ] {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
