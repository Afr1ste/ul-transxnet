#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential TN5000 ROI robustness probes.

Stage 1: light bbox jitter during classifier training.
Stage 2: predicted-box mix during classifier training, using detector boxes on
         the train split.

Each stage is one seed by default. The goal is to decide whether either path is
worth expanding to full 3-seed confirmation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import run_tn5000_compare_5models_3seed as runner


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(r"<LOCAL_CONDA_ROOT>\envs\Thyroid\python.exe")
LOG_ROOT = PROJECT_ROOT / "tn5000_roi_robustness_sequence_logs"
STATUS_PATH = LOG_ROOT / "tn5000_roi_robustness_sequence_latest.status.json"
DETECTOR_WEIGHTS = (
    PROJECT_ROOT
    / "tn5000_roi_detector_runs"
    / "yolo11n_tn5000_lesion_3seed_s27_20260502_130557"
    / "weights"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[17])
    p.add_argument("--num-epochs", type=int, default=70)
    p.add_argument("--detector-weights", default=str(DETECTOR_WEIGHTS))
    p.add_argument("--light-jitter-prob", type=float, default=0.25)
    p.add_argument("--light-jitter-center", type=float, default=0.04)
    p.add_argument("--light-jitter-scale", type=float, default=0.08)
    p.add_argument("--pred-box-prob", type=float, default=0.50)
    p.add_argument("--run-auto-roi-eval", type=int, default=1)
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


def list_subdirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    return [p for p in parent.iterdir() if p.is_dir()]


def newest_new_dir(parent: Path, before_names: Iterable[str]) -> Optional[Path]:
    before = set(before_names)
    new_dirs = [p for p in list_subdirs(parent) if p.name not in before]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    dirs = list_subdirs(parent)
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def base_common(args: argparse.Namespace) -> Dict:
    common = dict(runner.COMMON)
    common.update(dict(num_epochs=int(args.num_epochs)))
    common.pop("train_bbox_jitter_prob", None)
    common.pop("train_bbox_jitter_center", None)
    common.pop("train_bbox_jitter_scale", None)
    common.pop("train_pred_bbox_csv", None)
    common.pop("train_pred_bbox_prob", None)
    return common


def configure_runner_for_stage(
    args: argparse.Namespace,
    stage_name: str,
    output_root: str,
    log_root: str,
    tag: str,
    display: str,
    extra_common: Dict,
) -> None:
    common = base_common(args)
    common.update(extra_common)
    runner.OUTPUT_ROOT = output_root
    runner.LOG_ROOT = log_root
    runner.CONTINUE_ON_ERROR = False
    runner.SEEDS = list(args.seeds)
    runner.COMMON = common
    runner.BASE_CONFIGS = [
        dict(
            name=tag,
            display_name=display,
            model_family="custom",
            backbone_name="transxnet_t",
            backbone_module="models.transxnetggg",
            backbone_func="transxnet_t",
            backbone_out_dim=1000,
            **runner.COMMON,
        )
    ]
    runner.EXPERIMENTS = runner.build_experiments()
    update_status(current_stage=stage_name, current_runner_log_root=str(PROJECT_ROOT / log_root), current_output_root=str(PROJECT_ROOT / output_root))


def run_classifier_stage(args: argparse.Namespace, stage_name: str, output_root: str, log_root: str, tag: str, display: str, extra_common: Dict) -> Dict:
    configure_runner_for_stage(args, stage_name, output_root, log_root, tag, display, extra_common)
    log_root_path = PROJECT_ROOT / log_root
    before_logs = {p.name for p in list_subdirs(log_root_path)}
    update_status(phase=f"{stage_name}_training_running")
    rc = int(runner.main())
    classifier_log_dir = newest_new_dir(log_root_path, before_logs)
    result = {
        "stage": stage_name,
        "return_code": rc,
        "classifier_log_dir": str(classifier_log_dir) if classifier_log_dir else "",
    }
    update_status(phase=f"{stage_name}_training_done", **{f"{stage_name}_classifier_log_dir": result["classifier_log_dir"], f"{stage_name}_return_code": rc})
    return result


def run_auto_roi_eval(args: argparse.Namespace, stage_name: str, classifier_log_dir: Path, stamp: str) -> Dict:
    detector_weights = Path(args.detector_weights)
    out_dir = PROJECT_ROOT / "eval_reports" / f"tn5000_auto_roi_{stage_name}_{stamp}"
    cmd = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "eval_tn5000_auto_roi_pipeline.py"),
        "--detector-weights",
        str(detector_weights),
        "--classifier-log-dir",
        str(classifier_log_dir),
        "--output-dir",
        str(out_dir),
        "--splits",
        "val",
        "test",
        "--modes",
        "oracle",
        "auto",
        "full",
        "--device",
        "cuda",
    ]
    log_path = classifier_log_dir / f"auto_roi_eval_{stage_name}.log"
    update_status(phase=f"{stage_name}_auto_roi_eval_running", **{f"{stage_name}_auto_roi_eval_dir": str(out_dir)})
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        rc = int(proc.wait())
    update_status(phase=f"{stage_name}_auto_roi_eval_done", **{f"{stage_name}_auto_roi_eval_return_code": rc, f"{stage_name}_auto_roi_eval_log": str(log_path)})
    return {"return_code": rc, "output_dir": str(out_dir), "log_path": str(log_path)}


def export_train_boxes(args: argparse.Namespace, stamp: str) -> Path:
    detector_weights = Path(args.detector_weights)
    if not detector_weights.exists():
        raise FileNotFoundError(detector_weights)
    out_dir = PROJECT_ROOT / "eval_reports" / f"tn5000_detector_train_boxes_{stamp}"
    cmd = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "export_tn5000_detector_boxes.py"),
        "--detector-weights",
        str(detector_weights),
        "--output-dir",
        str(out_dir),
        "--splits",
        "train",
        "--device",
        "cuda",
    ]
    log_path = LOG_ROOT / f"export_train_boxes_{stamp}.log"
    update_status(phase="predbox_export_train_boxes_running", detector_train_boxes_dir=str(out_dir), detector_train_boxes_log=str(log_path))
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        rc = int(proc.wait())
    if rc != 0:
        update_status(phase="predbox_export_train_boxes_failed", detector_train_boxes_return_code=rc)
        raise RuntimeError("Detector train-box export failed: %s" % log_path)
    train_csv = out_dir / "detector_predictions" / "train_boxes.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    update_status(phase="predbox_export_train_boxes_done", detector_train_boxes_return_code=rc, detector_train_boxes_csv=str(train_csv))
    return train_csv


def worker(args: argparse.Namespace) -> int:
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(PYTHON_EXE)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    update_status(
        status="running",
        phase="worker_started",
        started_at=now(),
        stamp=stamp,
        seeds=args.seeds,
        num_epochs=args.num_epochs,
        detector_weights=str(Path(args.detector_weights)),
        light_jitter=dict(prob=args.light_jitter_prob, center=args.light_jitter_center, scale=args.light_jitter_scale),
        pred_box_prob=args.pred_box_prob,
    )

    stage_results: List[Dict] = []
    light_tag = "OURS_GGG_MCAON_bboxJITLIGHT_p%03d_c%03d_s%03d" % (
        int(round(args.light_jitter_prob * 100)),
        int(round(args.light_jitter_center * 1000)),
        int(round(args.light_jitter_scale * 1000)),
    )
    light = run_classifier_stage(
        args=args,
        stage_name="bboxjitter_light",
        output_root="tn5000_roi_runs_ggg_mca_bboxjitter_light",
        log_root="tn5000_ggg_mca_bboxjitter_light_logs",
        tag=light_tag,
        display="Ours-GGG-MCAON-lightBBoxJitter",
        extra_common=dict(
            train_bbox_jitter_prob=args.light_jitter_prob,
            train_bbox_jitter_center=args.light_jitter_center,
            train_bbox_jitter_scale=args.light_jitter_scale,
        ),
    )
    stage_results.append(light)
    if light["return_code"] != 0:
        update_status(status="failed", phase="bboxjitter_light_failed", stage_results=stage_results)
        return int(light["return_code"])
    if args.run_auto_roi_eval:
        eval_result = run_auto_roi_eval(args, "bboxjitter_light", Path(light["classifier_log_dir"]), stamp)
        light["auto_roi_eval"] = eval_result
        if eval_result["return_code"] != 0:
            update_status(status="failed", phase="bboxjitter_light_auto_roi_eval_failed", stage_results=stage_results)
            return int(eval_result["return_code"])

    train_boxes_csv = export_train_boxes(args, stamp)
    pred_tag = "OURS_GGG_MCAON_predBoxMix_p%03d" % int(round(args.pred_box_prob * 100))
    pred = run_classifier_stage(
        args=args,
        stage_name="predboxmix",
        output_root="tn5000_roi_runs_ggg_mca_predboxmix",
        log_root="tn5000_ggg_mca_predboxmix_logs",
        tag=pred_tag,
        display="Ours-GGG-MCAON-predBoxMix",
        extra_common=dict(
            train_pred_bbox_csv=str(train_boxes_csv),
            train_pred_bbox_prob=args.pred_box_prob,
        ),
    )
    stage_results.append(pred)
    if pred["return_code"] != 0:
        update_status(status="failed", phase="predboxmix_failed", stage_results=stage_results)
        return int(pred["return_code"])
    if args.run_auto_roi_eval:
        eval_result = run_auto_roi_eval(args, "predboxmix", Path(pred["classifier_log_dir"]), stamp)
        pred["auto_roi_eval"] = eval_result
        if eval_result["return_code"] != 0:
            update_status(status="failed", phase="predboxmix_auto_roi_eval_failed", stage_results=stage_results)
            return int(eval_result["return_code"])

    update_status(status="completed", phase="done", return_code=0, stage_results=stage_results)
    return 0


def launcher(args: argparse.Namespace) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_dir = LOG_ROOT / f"sequence_launcher_{stamp}"
    queue_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = queue_dir / "worker_stdout.log"
    stderr_log = queue_dir / "worker_stderr.log"
    cmd = [
        str(PYTHON_EXE),
        str(Path(__file__).resolve()),
        "--worker",
        "--seeds",
        *[str(s) for s in args.seeds],
        "--num-epochs",
        str(args.num_epochs),
        "--detector-weights",
        str(args.detector_weights),
        "--light-jitter-prob",
        str(args.light_jitter_prob),
        "--light-jitter-center",
        str(args.light_jitter_center),
        "--light-jitter-scale",
        str(args.light_jitter_scale),
        "--pred-box-prob",
        str(args.pred_box_prob),
        "--run-auto-roi-eval",
        str(args.run_auto_roi_eval),
    ]
    with stdout_log.open("wb") as out, stderr_log.open("wb") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=out,
            stderr=err,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    write_json(
        STATUS_PATH,
        {
            "status": "running",
            "phase": "detached_worker_started",
            "pid": proc.pid,
            "started_at": now(),
            "updated_at": now(),
            "queue_dir": str(queue_dir),
            "worker_stdout": str(stdout_log),
            "worker_stderr": str(stderr_log),
            "command": cmd,
        },
    )
    print("[LAUNCHED] pid=%s" % proc.pid)
    print("[STATUS] %s" % STATUS_PATH)
    print("[LOGDIR] %s" % queue_dir)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
