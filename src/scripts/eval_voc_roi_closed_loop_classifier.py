#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate trained BUSI/AUL ROI classifiers under oracle/auto/full inputs.

This script does not train classifiers. It reloads existing GGG-withMCA fold
checkpoints and evaluates:
  - oracle: original VOC annotations
  - auto: detector-predicted VOC annotations
  - full: whole image input, no ROI crop
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "busi": {
        "module": "fl_busi_roi_compare_5fold",
        "dataset_class": "BUSIVOCRoiDataset",
        "metrics_csv": PROJECT_ROOT / "busi_ggg_mca_clean_5fold_safe_logs" / "20260426_165332" / "all_runs_metrics.csv",
        "source_root": PROJECT_ROOT / "busi" / "busi_voc_v3_square_consistent",
        "default_ensemble_topk": 3,
        "default_bbox_expand_ratio": 0.30,
    },
    "aul": {
        "module": "fl_aul_roi_binary_compare_5fold",
        "dataset_class": "AULBinaryVOCRoiDataset",
        "metrics_csv": PROJECT_ROOT / "aul_ggg_mca_clean_5fold_safe_logs" / "20260426_200618" / "all_runs_metrics.csv",
        "source_root": PROJECT_ROOT / "aul" / "aul_voc_roi_v1",
        "default_ensemble_topk": 2,
        "default_bbox_expand_ratio": 0.20,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def boolish(value: str | bool | int, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default


def floatish(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def intish(row: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def top_checkpoints(run_dir: Path, topk: int, best_name: str) -> list[str]:
    scored: list[tuple[float, int, Path]] = []
    for path in run_dir.glob("epoch*_auc_*.pth"):
        m = re.search(r"epoch(\d+)_auc_([0-9.]+)\.pth$", path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        score = float(m.group(2))
        scored.append((score, epoch, path))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ckpts = [str(p) for _, _, p in scored[:topk]]
    if not ckpts:
        fallback = run_dir / best_name
        if fallback.exists():
            ckpts = [str(fallback)]
    return ckpts


def configure_module(mod, row: dict[str, str], data_root: Path, mode: str, dataset: str) -> None:
    cfg = mod.Config
    cfg.data_root = str(data_root)
    cfg.model_family = row.get("model_family") or row.get("model_family_cfg") or cfg.model_family
    cfg.backbone_name = row.get("backbone_name") or row.get("backbone_name_cfg") or cfg.backbone_name
    cfg.backbone_module = row.get("backbone_module") or row.get("backbone_module_cfg") or cfg.backbone_module
    cfg.backbone_func = row.get("backbone_func") or row.get("backbone_func_cfg") or cfg.backbone_func
    cfg.backbone_lr = floatish(row, "backbone_lr", floatish(row, "backbone_lr_cfg", cfg.backbone_lr))
    cfg.learning_rate = floatish(row, "head_lr", floatish(row, "head_lr_cfg", cfg.learning_rate))
    cfg.dropout = floatish(row, "dropout", cfg.dropout)
    cfg.weight_decay = floatish(row, "weight_decay", cfg.weight_decay)
    cfg.fold_idx = intish(row, "fold_idx", 0)
    cfg.train_split = f"fold{cfg.fold_idx}_train"
    cfg.val_split = f"fold{cfg.fold_idx}_val"
    cfg.test_split = "test"
    cfg.seed = intish(row, "seed", 17)
    cfg.input_size = intish(row, "input_size", cfg.input_size)
    cfg.batch_size = intish(row, "batch_size", cfg.batch_size)
    cfg.num_workers = 0
    cfg.bbox_expand_ratio = floatish(
        row,
        "bbox_expand_ratio",
        DATASET_CONFIGS[dataset]["default_bbox_expand_ratio"],
    )
    cfg.threshold_selection_mode = row.get("threshold_selection_mode") or cfg.threshold_selection_mode
    cfg.use_hflip_tta = boolish(row.get("use_hflip_tta", cfg.use_hflip_tta), cfg.use_hflip_tta)
    cfg.use_temperature_scaling = boolish(row.get("use_temperature_scaling", cfg.use_temperature_scaling), cfg.use_temperature_scaling)
    cfg.ensemble_topk = intish(row, "ensemble_topk_requested", DATASET_CONFIGS[dataset]["default_ensemble_topk"])
    cfg.use_roi_crop = mode != "full"


def make_loaders(mod, dataset_class, batch_size: int) -> dict[str, DataLoader]:
    _, eval_transform = mod.build_transforms()
    datasets = {
        "val": dataset_class(
            mod.Config.data_root,
            mod.Config.val_split,
            eval_transform,
            use_roi_crop=mod.Config.use_roi_crop,
            bbox_expand_ratio=mod.Config.bbox_expand_ratio,
            min_crop_size=mod.Config.min_crop_size,
            use_whole_image_fallback=mod.Config.use_whole_image_fallback,
        ),
        "test": dataset_class(
            mod.Config.data_root,
            mod.Config.test_split,
            eval_transform,
            use_roi_crop=mod.Config.use_roi_crop,
            bbox_expand_ratio=mod.Config.bbox_expand_ratio,
            min_crop_size=mod.Config.min_crop_size,
            use_whole_image_fallback=mod.Config.use_whole_image_fallback,
        ),
    }
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
        for split, ds in datasets.items()
    }


def metric_row(dataset: str, mode: str, fold: int, split: str, metrics: dict, threshold: float, temperature: float, used_ckpts: list[str], run_dir: Path, data_root: Path) -> dict[str, Any]:
    cm = metrics["cm"]
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "dataset": dataset,
        "mode": mode,
        "fold": fold,
        "split": split,
        "threshold": threshold,
        "temperature": temperature,
        "acc": metrics["acc"],
        "bal_acc": metrics["bal_acc"],
        "f1_macro": metrics["f1_macro"],
        "auc": metrics["auc"],
        "recall_0": recall_0,
        "recall_1": recall_1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(metrics["labels"])),
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "used_ckpts": "|".join(used_ckpts),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for mode in ["oracle", "auto", "full"]:
            for split in ["test"]:
                subset = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode and r["split"] == split]
                if not subset:
                    continue
                for metric in ["auc", "bal_acc", "f1_macro", "acc", "recall_0", "recall_1"]:
                    vals = [float(r[metric]) for r in subset]
                    s = stdev(vals) if len(vals) > 1 else 0.0
                    out.append(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "split": split,
                            "metric": metric,
                            "n_folds": len(vals),
                            "mean": mean(vals),
                            "std": s,
                            "mean_std": f"{mean(vals):.4f} +/- {s:.4f}",
                        }
                    )
    return out


def run_eval(dataset: str, mode: str, out_dir: Path, pred_voc_root_base: Path, device: torch.device) -> list[dict[str, Any]]:
    spec = DATASET_CONFIGS[dataset]
    mod = importlib.import_module(spec["module"])
    dataset_class = getattr(mod, spec["dataset_class"])
    rows = read_csv(spec["metrics_csv"])
    result_rows: list[dict[str, Any]] = []

    for row in rows:
        if row.get("status") != "ok":
            continue
        fold = intish(row, "fold_idx", 0)
        run_dir = Path(row["run_dir"])
        if mode == "oracle" or mode == "full":
            data_root = spec["source_root"]
        elif mode == "auto":
            data_root = pred_voc_root_base / f"{dataset}_fold{fold}_predbox"
        else:
            raise ValueError(mode)
        if not data_root.exists():
            raise FileNotFoundError(data_root)

        configure_module(mod, row, data_root=data_root, mode=mode, dataset=dataset)
        mod.set_seed(mod.Config.seed)
        loaders = make_loaders(mod, dataset_class, batch_size=mod.Config.batch_size)
        model = mod.UnifiedRoiClassifier(num_classes=mod.Config.num_classes).to(device)
        ckpts = top_checkpoints(run_dir, topk=mod.Config.ensemble_topk, best_name=mod.Config.best_model_name)
        if not ckpts:
            raise RuntimeError(f"No checkpoints found for {run_dir}")

        val_labels, val_logits, val_ids, val_paths, used_val = mod.collect_ensemble_logits(
            model, ckpts, loaders["val"], device, split_name=f"{dataset}-{mode}-fold{fold}-val", use_hflip_tta=mod.Config.use_hflip_tta
        )
        test_labels, test_logits, test_ids, test_paths, used_test = mod.collect_ensemble_logits(
            model, ckpts, loaders["test"], device, split_name=f"{dataset}-{mode}-fold{fold}-test", use_hflip_tta=mod.Config.use_hflip_tta
        )
        temperature = 1.0
        if mod.Config.use_temperature_scaling:
            temperature = mod.fit_temperature_on_val(val_logits, val_labels, device)
        val_probs = mod.logits_to_probs(val_logits, temperature=temperature)
        test_probs = mod.logits_to_probs(test_logits, temperature=temperature)
        threshold_results = mod.scan_thresholds(val_labels, val_probs, start=mod.Config.threshold_start, end=mod.Config.threshold_end, step=mod.Config.threshold_step)
        best_thr = mod.choose_best_threshold(threshold_results)
        threshold = float(best_thr["threshold"])
        val_metrics = mod.compute_metrics_from_labels_probs(val_labels, val_probs, threshold=threshold)
        test_metrics = mod.compute_metrics_from_labels_probs(test_labels, test_probs, threshold=threshold)

        fold_out = out_dir / dataset / mode / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        mod.save_threshold_scan_csv(threshold_results, fold_out / "val_threshold_scan.csv")
        mod.save_predictions_csv(val_ids, val_paths, val_metrics["labels"], val_metrics["preds"], val_metrics["probs"], threshold, fold_out / "val_predictions.csv")
        mod.save_predictions_csv(test_ids, test_paths, test_metrics["labels"], test_metrics["preds"], test_metrics["probs"], threshold, fold_out / "test_predictions.csv")
        (fold_out / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "fold": fold,
                    "run_dir": str(run_dir),
                    "data_root": str(data_root),
                    "used_ckpts": used_test,
                    "temperature": temperature,
                    "threshold": threshold,
                    "threshold_info": best_thr,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result_rows.append(metric_row(dataset, mode, fold, "val", val_metrics, threshold, temperature, used_test, run_dir, data_root))
        result_rows.append(metric_row(dataset, mode, fold, "test", test_metrics, threshold, temperature, used_test, run_dir, data_root))
        print(
            f"[{dataset}][{mode}][fold{fold}] test auc={test_metrics['auc']:.4f} "
            f"bal={test_metrics['bal_acc']:.4f} f1={test_metrics['f1_macro']:.4f} acc={test_metrics['acc']:.4f}"
        )
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["busi", "aul"], choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--modes", nargs="+", default=["oracle", "auto", "full"], choices=["oracle", "auto", "full"])
    parser.add_argument("--pred-voc-root-base", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_reports" / "busi_aul_closed_loop_auto_roi")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.parent / f"{args.out_dir.name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for mode in args.modes:
            all_rows.extend(run_eval(dataset, mode, out_dir=out_dir, pred_voc_root_base=args.pred_voc_root_base, device=device))
            write_csv(out_dir / "closed_loop_per_fold.csv", all_rows)
            write_csv(out_dir / "closed_loop_aggregate.csv", aggregate(all_rows))

    agg = aggregate(all_rows)
    md = ["# BUSI/AUL closed-loop ROI classification summary", ""]
    for dataset in args.datasets:
        md.append(f"## {dataset.upper()}")
        for mode in args.modes:
            md.append(f"### {mode}")
            for row in agg:
                if row["dataset"] == dataset and row["mode"] == mode:
                    md.append(f"- {row['metric']}: {row['mean_std']}")
            md.append("")
    md.extend(
        [
            "## Outputs",
            f"- per-fold CSV: `{out_dir / 'closed_loop_per_fold.csv'}`",
            f"- aggregate CSV: `{out_dir / 'closed_loop_aggregate.csv'}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
