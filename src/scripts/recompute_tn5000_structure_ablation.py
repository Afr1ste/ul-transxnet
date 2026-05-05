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
DEFAULT_ROOT = PROJECT_ROOT / "tn5000_paper_structure_ablation" / "20260405_132407"


@dataclass
class RunRecord:
    backbone_name: str
    seed: int
    result_dir: Path
    best_epoch: int
    best_score: float
    params: float
    thop_flops: float
    fvcore_flops: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute TN5000 structure ablation thresholds and aggregate tables from saved predictions."
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--threshold-start", type=float, default=0.10)
    p.add_argument("--threshold-end", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.01)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = load_rows(path)
    y_true = np.array([int(r["true_label"]) for r in rows], dtype=np.int64)
    prob1 = np.array([float(r["prob_class1"]) for r in rows], dtype=np.float64)
    return y_true, prob1


def compute_metrics(y_true: np.ndarray, prob1: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob1 >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    recall_0 = tn / (tn + fp) if (tn + fp) else float("nan")
    recall_1 = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "auc": float(roc_auc_score(y_true, prob1)),
        "recall_0": float(recall_0),
        "recall_1": float(recall_1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    n = int(round((end - start) / step)) + 1
    vals = np.round(start + np.arange(n) * step, 10)
    vals = vals[(vals >= start - 1e-12) & (vals <= end + 1e-12)]
    return vals


def scan_thresholds(y_true: np.ndarray, prob1: np.ndarray, start: float, end: float, step: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for thr in threshold_grid(start, end, step):
        metrics = compute_metrics(y_true, prob1, float(thr))
        rows.append(
            {
                "threshold": float(thr),
                "bal_acc": metrics["bal_acc"],
                "f1_macro": metrics["f1_macro"],
                "acc": metrics["acc"],
                "auc": metrics["auc"],
                "recall_0": metrics["recall_0"],
                "recall_1": metrics["recall_1"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    return rows


def choose_best_threshold(scan_rows: list[dict[str, float]]) -> float:
    best = max(scan_rows, key=lambda r: (r["bal_acc"], r["f1_macro"], r["acc"], r["threshold"]))
    return float(best["threshold"])


def parse_existing_runs(root: Path) -> list[RunRecord]:
    all_runs = load_rows(root / "structure_ablation_all_runs.csv")
    runs: list[RunRecord] = []
    for row in all_runs:
        runs.append(
            RunRecord(
                backbone_name=row["backbone_name"],
                seed=int(row["seed"]),
                result_dir=PROJECT_ROOT / row["result_dir"],
                best_epoch=int(row["best_epoch"]),
                best_score=float(row["best_score"]),
                params=float(row["params"]),
                thop_flops=float(row["thop_flops"]),
                fvcore_flops=float(row["fvcore_flops"]),
            )
        )
    return runs


def update_prediction_csv(path: Path, threshold: float) -> None:
    rows = load_rows(path)
    for row in rows:
        prob1 = float(row["prob_class1"])
        row["pred_label"] = str(int(prob1 >= threshold))
    save_rows(path, rows, fieldnames=list(rows[0].keys()))


def aggregate(all_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    order = ["transxnetggg", "transxnetgg", "transxnetg", "transxnet"]
    by_name: dict[str, list[dict[str, float]]] = {}
    for row in all_rows:
        by_name.setdefault(row["backbone_name"], []).append(row)
    out: list[dict[str, float]] = []
    for name in order:
        rows = by_name.get(name, [])
        if not rows:
            continue
        agg = {
            "backbone_name": name,
            "num_runs": len(rows),
        }
        for key in (
            "val_acc",
            "val_bal_acc",
            "val_f1_macro",
            "val_auc",
            "test_acc",
            "test_bal_acc",
            "test_f1_macro",
            "test_auc",
            "params",
            "thop_flops",
            "fvcore_flops",
        ):
            vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std(ddof=0))
        out.append(agg)
    return out


def maybe_update_summary(summary_path: Path, threshold: float, val_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
    text = summary_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text.split("\n", 2)[2])
    except Exception:
        return
    payload["selected_threshold"] = float(threshold)
    payload["val_metrics"].update(
        {
            "acc": float(val_metrics["acc"]),
            "bal_acc": float(val_metrics["bal_acc"]),
            "f1_macro": float(val_metrics["f1_macro"]),
            "auc": float(val_metrics["auc"]),
            "threshold": float(threshold),
        }
    )
    payload["test_metrics"].update(
        {
            "acc": float(test_metrics["acc"]),
            "bal_acc": float(test_metrics["bal_acc"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "auc": float(test_metrics["auc"]),
            "threshold": float(threshold),
        }
    )
    prefix = "TN5000 Unified Experiment Summary\n==========================================================================================\n"
    summary_path.write_text(prefix + json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    runs = parse_existing_runs(root)

    repaired_rows: list[dict[str, float]] = []
    repair_changes: list[dict[str, float]] = []
    for run in runs:
        val_path = run.result_dir / "val_predictions.csv"
        test_path = run.result_dir / "test_predictions.csv"
        scan_path = run.result_dir / "val_threshold_scan.csv"
        summary_path = run.result_dir / "summary.txt"

        y_val, p_val = load_predictions(val_path)
        y_test, p_test = load_predictions(test_path)
        scan_rows = scan_thresholds(y_val, p_val, args.threshold_start, args.threshold_end, args.threshold_step)
        best_thr = choose_best_threshold(scan_rows)
        val_metrics = compute_metrics(y_val, p_val, best_thr)
        test_metrics = compute_metrics(y_test, p_test, best_thr)

        save_rows(scan_path, scan_rows, fieldnames=list(scan_rows[0].keys()))
        update_prediction_csv(val_path, best_thr)
        update_prediction_csv(test_path, best_thr)
        maybe_update_summary(summary_path, best_thr, val_metrics, test_metrics)

        repaired_rows.append(
            {
                "backbone_name": run.backbone_name,
                "seed": run.seed,
                "result_dir": str(run.result_dir.relative_to(PROJECT_ROOT)),
                "best_epoch": run.best_epoch,
                "best_score": run.best_score,
                "selected_threshold": best_thr,
                "val_acc": val_metrics["acc"],
                "val_bal_acc": val_metrics["bal_acc"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_auc": val_metrics["auc"],
                "test_acc": test_metrics["acc"],
                "test_bal_acc": test_metrics["bal_acc"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_auc": test_metrics["auc"],
                "params": run.params,
                "thop_flops": run.thop_flops,
                "fvcore_flops": run.fvcore_flops,
            }
        )
        repair_changes.append(
            {
                "backbone_name": run.backbone_name,
                "seed": run.seed,
                "result_dir": str(run.result_dir.relative_to(PROJECT_ROOT)),
                "selected_threshold": best_thr,
                "val_bal_acc": val_metrics["bal_acc"],
                "test_bal_acc": test_metrics["bal_acc"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_auc": test_metrics["auc"],
            }
        )

    agg_rows = aggregate(repaired_rows)
    save_rows(root / "structure_ablation_all_runs.csv", repaired_rows, fieldnames=list(repaired_rows[0].keys()))
    save_rows(root / "structure_ablation_aggregate.csv", agg_rows, fieldnames=list(agg_rows[0].keys()))
    save_rows(root / "structure_ablation_threshold_repair.csv", repair_changes, fieldnames=list(repair_changes[0].keys()))
    (root / "structure_ablation_threshold_repair_manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "root": str(root),
                "threshold_start": args.threshold_start,
                "threshold_end": args.threshold_end,
                "threshold_step": args.threshold_step,
                "num_runs": len(repaired_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] repaired TN5000 structure ablation under {root}")


if __name__ == "__main__":
    main()
