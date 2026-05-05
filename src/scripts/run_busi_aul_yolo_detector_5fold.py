#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run BUSI/AUL one-class lesion detector 5-fold queues.

This launcher starts a detached worker so the queue can continue after the
interactive session is idle. Each fold trains a diagnosis-agnostic YOLO detector
with one class: lesion. After training, the best checkpoint is validated on the
held-out test split defined in the corresponding fold YAML.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(r"C:\Users\Afr1ste\anaconda3\envs\Thyroid\python.exe")
YOLO_EXE = Path(r"C:\Users\Afr1ste\anaconda3\envs\Thyroid\Scripts\yolo.exe")

DATASETS = {
    "busi": PROJECT_ROOT / "detector_datasets" / "busi_yolo_lesion_v1",
    "aul": PROJECT_ROOT / "detector_datasets" / "aul_yolo_lesion_v1",
}

RUN_ROOT = PROJECT_ROOT / "busi_aul_roi_detector_runs"
LOG_ROOT = PROJECT_ROOT / "busi_aul_roi_detector_logs"
STATUS_PATH = LOG_ROOT / "busi_aul_yolo_detector_5fold_latest.status.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--datasets", nargs="+", default=["busi", "aul"], choices=sorted(DATASETS))
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--model", default="yolo11n.pt")
    return p.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_status() -> Dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def update_status(**updates) -> None:
    status = load_status()
    status.update(updates)
    status["updated_at"] = now()
    write_json(STATUS_PATH, status)


def run_command(cmd: List[str], cwd: Path, stdout_log: Path, stderr_log: Path) -> int:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("wb") as out, stderr_log.open("wb") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err)
        return int(proc.wait())


def parse_best_train_metrics(run_dir: Path) -> Dict:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def value(row: Dict, key: str) -> float:
        try:
            return float(str(row.get(key, "nan")).strip())
        except ValueError:
            return float("nan")

    best = max(rows, key=lambda r: value(r, "metrics/mAP50-95(B)"))
    return {
        "best_epoch_by_map50_95": int(float(best.get("epoch", 0))),
        "val_precision": value(best, "metrics/precision(B)"),
        "val_recall": value(best, "metrics/recall(B)"),
        "val_map50": value(best, "metrics/mAP50(B)"),
        "val_map50_95": value(best, "metrics/mAP50-95(B)"),
    }


def parse_metrics_from_log(log_path: Path) -> Dict:
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    candidates = [line for line in lines if line.strip().startswith("all ")]
    if not candidates:
        return {}
    parts = candidates[-1].split()
    nums = []
    for token in parts:
        try:
            nums.append(float(token))
        except ValueError:
            pass
    if len(nums) < 6:
        return {}
    return {
        "test_images": int(nums[0]),
        "test_instances": int(nums[1]),
        "test_precision": float(nums[-4]),
        "test_recall": float(nums[-3]),
        "test_map50": float(nums[-2]),
        "test_map50_95": float(nums[-1]),
    }


def append_summary_row(summary_csv: Path, row: Dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    fieldnames = [
        "dataset",
        "fold",
        "run_name",
        "return_code",
        "train_run_dir",
        "best_weights",
        "best_epoch_by_map50_95",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_images",
        "test_instances",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
    ]
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_aggregate(summary_csv: Path, aggregate_csv: Path, summary_md: Path) -> None:
    rows = read_summary_rows(summary_csv)
    aggregate_rows: list[dict[str, object]] = []
    md_lines = ["# BUSI/AUL YOLO lesion detector 5-fold summary", ""]
    for dataset in sorted({row["dataset"] for row in rows}):
        ds_rows = [row for row in rows if row.get("dataset") == dataset and str(row.get("return_code")) == "0"]
        md_lines.append(f"## {dataset.upper()}")
        if not ds_rows:
            md_lines.append("")
            md_lines.append("No completed folds.")
            md_lines.append("")
            continue
        for metric in ["val_map50", "val_map50_95", "test_map50", "test_map50_95", "test_precision", "test_recall"]:
            vals = []
            for row in ds_rows:
                try:
                    vals.append(float(row[metric]))
                except (KeyError, ValueError):
                    pass
            if not vals:
                continue
            s = stdev(vals) if len(vals) > 1 else 0.0
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_folds": len(vals),
                    "mean": mean(vals),
                    "std": s,
                    "mean_std": f"{mean(vals):.4f} +/- {s:.4f}",
                }
            )
            md_lines.append(f"- {metric}: {mean(vals):.4f} +/- {s:.4f}")
        md_lines.append("")

    if aggregate_rows:
        aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["dataset", "metric", "n_folds", "mean", "std", "mean_std"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")


def launcher(args: argparse.Namespace) -> int:
    if not YOLO_EXE.exists():
        raise FileNotFoundError(YOLO_EXE)
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(PYTHON_EXE)
    for dataset in args.datasets:
        data_root = DATASETS[dataset]
        for fold in args.folds:
            data_yaml = data_root / f"data_fold{fold}.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(data_yaml)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"busi_aul_yolo_5fold_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_stdout = log_dir / "worker_stdout.log"
    worker_stderr = log_dir / "worker_stderr.log"

    cmd = [
        str(PYTHON_EXE),
        str(Path(__file__).resolve()),
        "--worker",
        "--datasets",
        *list(args.datasets),
        "--folds",
        *[str(fold) for fold in args.folds],
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--model",
        str(args.model),
    ]

    with worker_stdout.open("wb") as out, worker_stderr.open("wb") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=out,
            stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    status = {
        "status": "running",
        "phase": "detached_worker_started",
        "pid": proc.pid,
        "started_at": now(),
        "stamp": stamp,
        "datasets": args.datasets,
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "log_dir": str(log_dir),
        "worker_stdout": str(worker_stdout),
        "worker_stderr": str(worker_stderr),
        "summary_csv": str(log_dir / "detector_5fold_summary.csv"),
        "aggregate_csv": str(log_dir / "detector_5fold_aggregate.csv"),
        "summary_md": str(log_dir / "summary.md"),
        "command": cmd,
    }
    write_json(STATUS_PATH, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def worker(args: argparse.Namespace) -> int:
    status = load_status()
    log_dir = Path(status.get("log_dir", LOG_ROOT / ("busi_aul_yolo_5fold_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    summary_csv = Path(status.get("summary_csv", log_dir / "detector_5fold_summary.csv"))
    aggregate_csv = Path(status.get("aggregate_csv", log_dir / "detector_5fold_aggregate.csv"))
    summary_md = Path(status.get("summary_md", log_dir / "summary.md"))
    completed: list[dict[str, object]] = []
    update_status(status="running", phase="queue_started", completed=[])

    for dataset in args.datasets:
        data_root = DATASETS[dataset]
        for fold in args.folds:
            run_name = f"yolo11n_{dataset}_lesion_f{fold}_{status.get('stamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}"
            train_run_dir = RUN_ROOT / dataset / run_name
            best_weights = train_run_dir / "weights" / "best.pt"
            data_yaml = data_root / f"data_fold{fold}.yaml"
            fold_log_dir = log_dir / dataset / f"fold{fold}"
            train_stdout = fold_log_dir / "train_stdout.log"
            train_stderr = fold_log_dir / "train_stderr.log"
            test_stdout = fold_log_dir / "test_val_stdout.log"
            test_stderr = fold_log_dir / "test_val_stderr.log"

            train_cmd = [
                str(YOLO_EXE),
                "detect",
                "train",
                f"data={data_yaml}",
                f"model={args.model}",
                f"epochs={args.epochs}",
                f"imgsz={args.imgsz}",
                f"batch={args.batch}",
                f"device={args.device}",
                f"workers={args.workers}",
                f"patience={args.patience}",
                f"seed={17 + fold}",
                "cache=False",
                f"project={RUN_ROOT / dataset}",
                f"name={run_name}",
                "exist_ok=False",
            ]
            update_status(
                phase="training_detector",
                current_dataset=dataset,
                current_fold=fold,
                current_run_dir=str(train_run_dir),
                current_stdout=str(train_stdout),
                current_stderr=str(train_stderr),
                current_command=train_cmd,
            )
            rc = run_command(train_cmd, PROJECT_ROOT, train_stdout, train_stderr)
            train_metrics = parse_best_train_metrics(train_run_dir)

            test_metrics: Dict = {}
            if rc == 0 and best_weights.exists():
                test_cmd = [
                    str(YOLO_EXE),
                    "detect",
                    "val",
                    f"model={best_weights}",
                    f"data={data_yaml}",
                    "split=test",
                    f"imgsz={args.imgsz}",
                    f"batch={args.batch}",
                    f"device={args.device}",
                    f"workers={args.workers}",
                    f"project={RUN_ROOT / dataset}",
                    f"name={run_name}_testval",
                    "exist_ok=False",
                ]
                update_status(phase="validating_detector_test", current_dataset=dataset, current_fold=fold, current_command=test_cmd)
                test_rc = run_command(test_cmd, PROJECT_ROOT, test_stdout, test_stderr)
                test_metrics = parse_metrics_from_log(test_stdout)
                if test_rc != 0:
                    test_metrics["test_val_return_code"] = test_rc
            else:
                test_metrics["test_val_skipped"] = True

            row = {
                "dataset": dataset,
                "fold": fold,
                "run_name": run_name,
                "return_code": rc,
                "train_run_dir": str(train_run_dir),
                "best_weights": str(best_weights) if best_weights.exists() else "",
                **train_metrics,
                **test_metrics,
            }
            completed.append({"dataset": dataset, "fold": fold, "return_code": rc})
            append_summary_row(summary_csv, row)
            write_aggregate(summary_csv, aggregate_csv, summary_md)
            update_status(
                phase="fold_completed" if rc == 0 else "fold_failed",
                current_dataset=dataset,
                current_fold=fold,
                completed=completed,
                last_row=row,
                summary_csv=str(summary_csv),
                aggregate_csv=str(aggregate_csv),
                summary_md=str(summary_md),
            )
            if rc != 0:
                update_status(status="failed", phase="training_detector_failed", failed_dataset=dataset, failed_fold=fold)
                return rc

    update_status(status="completed", phase="queue_completed", current_dataset=None, current_fold=None, completed=completed)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
