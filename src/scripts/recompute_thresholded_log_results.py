from __future__ import annotations

import argparse
import csv
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute thresholded metrics and aggregated tables from existing prediction CSVs."
    )
    p.add_argument("--module", required=True, help="Runner module, e.g. run_aul_roi_binary_compare_5models_5fold")
    p.add_argument("--log-dir", required=True, help="Existing log directory to repair")
    p.add_argument("--threshold-start", type=float, default=0.10)
    p.add_argument("--threshold-end", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.01)
    return p.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_boundary_threshold(threshold: float, start: float, end: float, step: float) -> bool:
    tol = max(step * 0.51, 1e-9)
    return abs(float(threshold) - float(start)) <= tol or abs(float(threshold) - float(end)) <= tol


def refresh_single_run(
    runner_module,
    run_row: dict[str, Any],
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> dict[str, Any]:
    run_dir = Path(run_row["run_dir"])
    val_pred_path = run_dir / "val_predictions.csv"
    test_pred_path = run_dir / "test_predictions.csv"

    if not val_pred_path.exists() or not test_pred_path.exists():
        raise FileNotFoundError(f"Missing predictions under {run_dir}")

    old_threshold = float(run_row.get("selected_threshold", 0.5))

    val_rows = runner_module.load_predictions_csv(val_pred_path)
    test_rows = runner_module.load_predictions_csv(test_pred_path)
    threshold_rows = runner_module.scan_thresholds(
        val_rows,
        start=threshold_start,
        end=threshold_end,
        step=threshold_step,
    )
    best_thr_info = runner_module.choose_best_threshold(threshold_rows)
    best_threshold = float(best_thr_info["threshold"])

    runner_module.write_rows_csv(
        run_dir / "val_threshold_scan.csv",
        threshold_rows,
        fieldnames=list(threshold_rows[0].keys()),
    )
    runner_module.save_predictions_with_metrics(val_rows, best_threshold, val_pred_path)
    runner_module.save_predictions_with_metrics(test_rows, best_threshold, test_pred_path)

    val_metrics = runner_module.compute_metrics_from_predictions(runner_module.load_predictions_csv(val_pred_path))
    test_metrics = runner_module.compute_metrics_from_predictions(runner_module.load_predictions_csv(test_pred_path))

    updated = dict(run_row)
    updated["selected_threshold"] = best_threshold
    updated["threshold_boundary_hit"] = int(
        is_boundary_threshold(best_threshold, threshold_start, threshold_end, threshold_step)
    )

    updated["val_acc"] = val_metrics["acc"]
    updated["val_bal_acc"] = val_metrics["bal_acc"]
    updated["val_f1_macro"] = val_metrics["f1_macro"]
    updated["val_auc"] = val_metrics["auc"]
    updated["val_recall_0"] = val_metrics["recall_0"]
    updated["val_recall_1"] = val_metrics["recall_1"]
    updated["val_tn"] = val_metrics["tn"]
    updated["val_fp"] = val_metrics["fp"]
    updated["val_fn"] = val_metrics["fn"]
    updated["val_tp"] = val_metrics["tp"]

    updated["test_acc"] = test_metrics["acc"]
    updated["test_bal_acc"] = test_metrics["bal_acc"]
    updated["test_f1_macro"] = test_metrics["f1_macro"]
    updated["test_auc"] = test_metrics["auc"]
    updated["test_recall_0"] = test_metrics["recall_0"]
    updated["test_recall_1"] = test_metrics["recall_1"]
    updated["test_tn"] = test_metrics["tn"]
    updated["test_fp"] = test_metrics["fp"]
    updated["test_fn"] = test_metrics["fn"]
    updated["test_tp"] = test_metrics["tp"]

    updated["_repair_old_threshold"] = old_threshold
    updated["_repair_new_threshold"] = best_threshold
    return updated


def detect_model_level_threshold_file(model_dir: Path) -> Path:
    for name in ("val_oof_threshold_scan.csv", "val_ensemble_threshold_scan.csv"):
        path = model_dir / name
        if path.exists():
            return path
    return model_dir / "val_threshold_scan.csv"


def collect_model_level_summary(log_dir: Path, threshold_start: float, threshold_end: float, threshold_step: float) -> list[dict[str, Any]]:
    model_level_root = log_dir / "model_level"
    rows: list[dict[str, Any]] = []
    if not model_level_root.exists():
        return rows

    for model_dir in sorted(p for p in model_level_root.iterdir() if p.is_dir()):
        metrics_path = model_dir / "test_ensemble_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = load_csv_rows(metrics_path)[0]
        rows.append(
            {
                "model_dir": model_dir.name,
                "threshold": float(metrics["threshold"]),
                "boundary_hit": int(
                    is_boundary_threshold(
                        float(metrics["threshold"]),
                        threshold_start,
                        threshold_end,
                        threshold_step,
                    )
                ),
                "auc": float(metrics["auc"]),
                "bal_acc": float(metrics["bal_acc"]),
                "f1_macro": float(metrics["f1_macro"]),
                "acc": float(metrics["acc"]),
                "threshold_scan_file": str(detect_model_level_threshold_file(model_dir).name),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    runner_module = importlib.import_module(args.module)
    log_dir = Path(args.log_dir)

    batch_status_path = log_dir / "batch_status.csv"
    all_runs_path = log_dir / "all_runs_metrics.csv"
    if not batch_status_path.exists() or not all_runs_path.exists():
        raise FileNotFoundError(f"Missing batch_status.csv or all_runs_metrics.csv under {log_dir}")

    statuses = load_csv_rows(batch_status_path)
    run_rows = load_csv_rows(all_runs_path)

    repaired_rows: list[dict[str, Any]] = []
    threshold_changes: list[dict[str, Any]] = []
    for row in run_rows:
        if str(row.get("status", "")).lower() != "ok":
            repaired_rows.append(row)
            continue
        updated = refresh_single_run(
            runner_module,
            row,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
        )
        threshold_changes.append(
            {
                "name": updated.get("name", ""),
                "base_name": updated.get("base_name", ""),
                "run_dir": updated.get("run_dir", ""),
                "old_threshold": updated.pop("_repair_old_threshold"),
                "new_threshold": updated.pop("_repair_new_threshold"),
                "boundary_hit": updated.get("threshold_boundary_hit", 0),
            }
        )
        repaired_rows.append(updated)

    runner_module.save_incremental_outputs(log_dir, statuses, repaired_rows)
    runner_module.build_model_level_artifacts(log_dir, repaired_rows)

    model_level_rows = collect_model_level_summary(
        log_dir,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
    )
    if model_level_rows:
        runner_module.write_rows_csv(
            log_dir / "threshold_repair_model_level_summary.csv",
            model_level_rows,
            fieldnames=list(model_level_rows[0].keys()),
        )
    if threshold_changes:
        runner_module.write_rows_csv(
            log_dir / "threshold_repair_changes.csv",
            threshold_changes,
            fieldnames=list(threshold_changes[0].keys()),
        )

    save_json(
        log_dir / "threshold_repair_manifest.json",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "module": args.module,
            "log_dir": str(log_dir),
            "threshold_start": args.threshold_start,
            "threshold_end": args.threshold_end,
            "threshold_step": args.threshold_step,
            "num_runs": len(run_rows),
            "num_repaired_runs": len(threshold_changes),
            "num_boundary_hits_model_level": sum(int(r["boundary_hit"]) for r in model_level_rows),
        },
    )

    print(f"[DONE] repaired log dir: {log_dir}")
    print(f"[DONE] module: {args.module}")
    print(f"[DONE] repaired runs: {len(threshold_changes)}")
    if model_level_rows:
        print(f"[DONE] model-level summary: {log_dir / 'threshold_repair_model_level_summary.csv'}")


if __name__ == "__main__":
    main()
