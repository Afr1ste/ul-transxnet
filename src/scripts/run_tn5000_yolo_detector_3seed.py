#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a 3-seed TN5000 one-class lesion detector queue.

The queue trains YOLO11n detectors for automatic ROI localization. It is kept
diagnosis-agnostic: detector labels contain only one class, "lesion".

By default, each detector is followed by:
  1) an independent YOLO validation on the TN5000 test split;
  2) an auto-ROI classification evaluation using the existing UL-TransXNet
     checkpoint ensemble.

The launcher starts a detached worker and writes a status JSON so progress can
be checked without keeping this terminal attached.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(r"C:\Users\Afr1ste\anaconda3\envs\Thyroid\python.exe")
YOLO_EXE = Path(r"C:\Users\Afr1ste\anaconda3\envs\Thyroid\Scripts\yolo.exe")
DATA_YAML = PROJECT_ROOT / "detector_datasets" / "tn5000_yolo_lesion_v1" / "data.yaml"
RUN_ROOT = PROJECT_ROOT / "tn5000_roi_detector_runs"
LOG_ROOT = PROJECT_ROOT / "tn5000_roi_detector_logs"
EVAL_SCRIPT = PROJECT_ROOT / "eval_tn5000_auto_roi_pipeline.py"
STATUS_PATH = LOG_ROOT / "tn5000_yolo_detector_3seed_latest.status.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true", help="Internal flag used by the detached queue worker.")
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 27, 37])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--run-auto-roi-eval", type=int, default=1)
    p.add_argument("--classifier-log-dir", default=r"tn5000_ggg_mca_enabled_3seed_logs\20260426_093728")
    p.add_argument("--eval-modes", nargs="+", default=["auto"], choices=["oracle", "auto", "full"])
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
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
        return float(str(row.get(key, "nan")).strip())
    best = max(rows, key=lambda r: value(r, "metrics/mAP50-95(B)"))
    return {
        "best_epoch_by_map50_95": int(float(best.get("epoch", 0))),
        "val_precision": value(best, "metrics/precision(B)"),
        "val_recall": value(best, "metrics/recall(B)"),
        "val_map50": value(best, "metrics/mAP50(B)"),
        "val_map50_95": value(best, "metrics/mAP50-95(B)"),
    }


def parse_metrics_from_log(log_path: Path) -> Dict:
    """Best-effort parse of the final Ultralytics val row from stdout."""
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    candidates = [line for line in lines if line.strip().startswith("all ")]
    if not candidates:
        return {}
    parts = candidates[-1].split()
    # Expected tail: all images instances Box(P R mAP50 mAP50-95)
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
        "seed",
        "run_name",
        "return_code",
        "train_run_dir",
        "best_weights",
        "val_map50",
        "val_map50_95",
        "test_map50",
        "test_map50_95",
        "auto_roi_eval_dir",
    ]
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def launcher(args: argparse.Namespace) -> int:
    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)
    if not YOLO_EXE.exists():
        raise FileNotFoundError(YOLO_EXE)
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(PYTHON_EXE)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / f"tn5000_yolo_3seed_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_stdout = log_dir / "worker_stdout.log"
    worker_stderr = log_dir / "worker_stderr.log"

    cmd = [
        str(PYTHON_EXE),
        str(Path(__file__).resolve()),
        "--worker",
        "--seeds",
        *[str(s) for s in args.seeds],
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
        "--run-auto-roi-eval",
        str(args.run_auto_roi_eval),
        "--classifier-log-dir",
        str(args.classifier_log_dir),
        "--eval-modes",
        *list(args.eval_modes),
        "--eval-splits",
        *list(args.eval_splits),
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
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "data_yaml": str(DATA_YAML),
        "log_dir": str(log_dir),
        "worker_stdout": str(worker_stdout),
        "worker_stderr": str(worker_stderr),
        "summary_csv": str(log_dir / "detector_3seed_summary.csv"),
        "command": cmd,
    }
    write_json(STATUS_PATH, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def worker(args: argparse.Namespace) -> int:
    status = load_status()
    log_dir = Path(status.get("log_dir", LOG_ROOT / ("tn5000_yolo_3seed_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    summary_csv = Path(status.get("summary_csv", log_dir / "detector_3seed_summary.csv"))
    rows = []
    update_status(status="running", phase="queue_started", current_seed=None, completed_seeds=[])

    for seed in args.seeds:
        run_name = f"yolo11n_tn5000_lesion_3seed_s{seed}_{status.get('stamp', datetime.now().strftime('%Y%m%d_%H%M%S'))}"
        train_run_dir = RUN_ROOT / run_name
        best_weights = train_run_dir / "weights" / "best.pt"
        seed_log_dir = log_dir / f"seed_{seed}"
        train_stdout = seed_log_dir / "train_stdout.log"
        train_stderr = seed_log_dir / "train_stderr.log"
        test_stdout = seed_log_dir / "test_val_stdout.log"
        test_stderr = seed_log_dir / "test_val_stderr.log"
        eval_stdout = seed_log_dir / "auto_roi_eval_stdout.log"
        eval_stderr = seed_log_dir / "auto_roi_eval_stderr.log"
        auto_roi_eval_dir = PROJECT_ROOT / "eval_reports" / f"tn5000_auto_roi_yolo11n_3seed_s{seed}_{status.get('stamp', '')}"

        train_cmd = [
            str(YOLO_EXE),
            "detect",
            "train",
            f"data={DATA_YAML}",
            f"model={args.model}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"device={args.device}",
            f"workers={args.workers}",
            f"patience={args.patience}",
            f"seed={seed}",
            "cache=False",
            f"project={RUN_ROOT}",
            f"name={run_name}",
            "exist_ok=False",
        ]
        update_status(
            phase="training_detector",
            current_seed=seed,
            current_run_dir=str(train_run_dir),
            current_stdout=str(train_stdout),
            current_stderr=str(train_stderr),
            current_command=train_cmd,
        )
        rc = run_command(train_cmd, PROJECT_ROOT, train_stdout, train_stderr)
        train_metrics = parse_best_train_metrics(train_run_dir)

        test_metrics = {}
        if rc == 0 and best_weights.exists():
            test_cmd = [
                str(YOLO_EXE),
                "detect",
                "val",
                f"model={best_weights}",
                f"data={DATA_YAML}",
                "split=test",
                f"imgsz={args.imgsz}",
                f"batch={args.batch}",
                f"device={args.device}",
                f"workers={args.workers}",
                f"project={RUN_ROOT}",
                f"name={run_name}_testval",
                "exist_ok=False",
            ]
            update_status(phase="validating_detector_test", current_seed=seed, current_command=test_cmd)
            test_rc = run_command(test_cmd, PROJECT_ROOT, test_stdout, test_stderr)
            test_metrics = parse_metrics_from_log(test_stdout)
            if test_rc != 0:
                test_metrics["test_val_return_code"] = test_rc
        else:
            test_metrics["test_val_skipped"] = True

        eval_rc = None
        if rc == 0 and best_weights.exists() and int(args.run_auto_roi_eval):
            eval_cmd = [
                str(PYTHON_EXE),
                str(EVAL_SCRIPT),
                "--detector-weights",
                str(best_weights),
                "--classifier-log-dir",
                str(args.classifier_log_dir),
                "--output-dir",
                str(auto_roi_eval_dir),
                "--splits",
                *list(args.eval_splits),
                "--modes",
                *list(args.eval_modes),
                "--device",
                "cuda" if str(args.device) != "cpu" else "cpu",
            ]
            update_status(phase="evaluating_auto_roi_classifier", current_seed=seed, current_command=eval_cmd)
            eval_rc = run_command(eval_cmd, PROJECT_ROOT, eval_stdout, eval_stderr)

        row = {
            "seed": seed,
            "run_name": run_name,
            "return_code": rc,
            "train_run_dir": str(train_run_dir),
            "best_weights": str(best_weights) if best_weights.exists() else "",
            "auto_roi_eval_dir": str(auto_roi_eval_dir) if auto_roi_eval_dir.exists() else "",
            **train_metrics,
            **test_metrics,
            "auto_roi_eval_return_code": eval_rc,
        }
        rows.append(row)
        append_summary_row(summary_csv, row)
        update_status(
            phase="seed_completed" if rc == 0 else "seed_failed",
            current_seed=seed,
            completed_seeds=[r["seed"] for r in rows if r.get("return_code") == 0],
            last_row=row,
            summary_csv=str(summary_csv),
        )
        if rc != 0:
            update_status(status="failed", phase="training_detector_failed", failed_seed=seed)
            return rc

    update_status(status="completed", phase="queue_completed", current_seed=None, completed_seeds=args.seeds)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())

