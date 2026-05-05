#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-only TN5000 ROI crop-rule sweep.

This script reuses existing UL-TransXNet checkpoints and cached detector boxes
to compare ROI expansion ratios and rectangular-vs-square crop geometry. It is
GPU-heavy because each crop rule changes classifier inputs and therefore logits.

Recommended first pass:
  --modes oracle auto --geometries rect square --expand-ratios 0.0 0.2 0.3 0.4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

import eval_tn5000_auto_roi_pipeline as pipe


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tn5000-root", default=r"C:\Users\Afr1ste\PycharmProjects\Thyroid\TN5000_forReview")
    p.add_argument("--detector-pipeline-dir", default=r"eval_reports\tn5000_auto_roi_pipeline_yolo11n_20260502_1245")
    p.add_argument("--classifier-log-dir", default=r"tn5000_ggg_mca_enabled_3seed_logs\20260426_093728")
    p.add_argument("--output-dir", default="")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--modes", nargs="+", default=["oracle", "auto"], choices=["oracle", "auto", "full"])
    p.add_argument("--geometries", nargs="+", default=["rect", "square"], choices=["rect", "square"])
    p.add_argument("--expand-ratios", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--classifier-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--min-crop-size", type=int, default=64)
    p.add_argument("--hflip-tta", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-configs", type=int, default=0, help="Optional debug cap.")
    return p.parse_args()


def load_detector_preds(pipeline_dir: Path, splits: Sequence[str]) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for split in splits:
        path = pipeline_dir / "detector_predictions" / f"{split}_boxes.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = pipe.csv.DictReader(path.open("r", encoding="utf-8", newline=""))
        split_preds: Dict[str, Dict] = {}
        for row in rows:
            split_preds[row["image_id"]] = {
                "pred_bbox": pipe.parse_box_string(row.get("pred_bbox", "")),
                "pred_conf": float(row.get("pred_conf", 0.0)),
                "iou_gt": float(row.get("iou_gt", 0.0)),
                "no_detection": str(row.get("no_detection", "0")).lower() in {"1", "true", "yes"},
            }
        out[split] = split_preds
    return out


def make_configs(args: argparse.Namespace) -> List[Dict]:
    configs = []
    for mode in args.modes:
        if mode == "full":
            configs.append({"mode": mode, "geometry": "full", "expand_ratio": 0.0, "square_crop": False})
            continue
        for geometry in args.geometries:
            for ratio in args.expand_ratios:
                configs.append(
                    {
                        "mode": mode,
                        "geometry": geometry,
                        "expand_ratio": float(ratio),
                        "square_crop": geometry == "square",
                    }
                )
    if args.max_configs and args.max_configs > 0:
        return configs[: args.max_configs]
    return configs


def main() -> int:
    args = parse_args()
    root = Path(args.tn5000_root)
    pipeline_dir = Path(args.detector_pipeline_dir)
    if not pipeline_dir.is_absolute():
        pipeline_dir = PROJECT_ROOT / pipeline_dir
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "eval_reports" / ("tn5000_roi_crop_rule_sweep_" + time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    pipe.configure_classifier(args.input_size)
    transform = pipe.build_eval_transform(args.input_size)
    samples_by_split = {split: pipe.load_samples(root, split) for split in args.splits}
    detector_preds = load_detector_preds(pipeline_dir, args.splits)
    ckpts = pipe.load_classifier_ckpts(Path(args.classifier_log_dir))
    pipe.write_json(out_dir / "classifier_ckpts_used.json", [str(p) for p in ckpts])

    rows: List[Dict] = []
    configs = make_configs(args)
    pipe.write_json(
        out_dir / "run_config.json",
        {
            "tn5000_root": str(root),
            "detector_pipeline_dir": str(pipeline_dir),
            "classifier_log_dir": args.classifier_log_dir,
            "splits": args.splits,
            "configs": configs,
            "device": str(device),
        },
    )

    for idx, cfg in enumerate(configs, start=1):
        logits_by_split = {}
        labels_by_split = {}
        print(f"[CONFIG {idx}/{len(configs)}] {cfg}")
        for split in args.splits:
            ds = pipe.AutoRoiClassificationDataset(
                samples_by_split[split],
                detector_preds.get(split, {}),
                mode=cfg["mode"],
                transform=transform,
                expand_ratio=float(cfg["expand_ratio"]),
                min_crop_size=args.min_crop_size,
                square_crop=bool(cfg["square_crop"]),
            )
            loader = DataLoader(ds, batch_size=args.classifier_batch_size, shuffle=False, num_workers=args.num_workers)
            labels, logits, _, _, _ = pipe.collect_ensemble_logits(
                ckpts,
                loader,
                device=device,
                hflip_tta=bool(args.hflip_tta),
                split=split,
                mode=f"{cfg['mode']}_{cfg['geometry']}_e{cfg['expand_ratio']:.2f}",
            )
            logits_by_split[split] = logits
            labels_by_split[split] = labels

        if "val" not in logits_by_split:
            raise RuntimeError("val split is required for threshold selection")
        temperature = pipe.fit_temperature(logits_by_split["val"], labels_by_split["val"], device)
        val_probs = pipe.logits_to_probs(logits_by_split["val"], temperature)
        threshold_rows = pipe.scan_thresholds(labels_by_split["val"], val_probs)
        threshold = float(threshold_rows[0]["threshold"])

        for split in args.splits:
            probs = pipe.logits_to_probs(logits_by_split[split], temperature)
            metrics = pipe.compute_metrics(labels_by_split[split], probs, threshold)
            metrics.update(
                {
                    "mode": cfg["mode"],
                    "geometry": cfg["geometry"],
                    "expand_ratio": float(cfg["expand_ratio"]),
                    "square_crop": int(bool(cfg["square_crop"])),
                    "split": split,
                    "temperature": float(temperature),
                }
            )
            rows.append(metrics)
            pipe.write_csv(out_dir / "roi_crop_rule_sweep_metrics.csv", rows)

    pipe.write_csv(out_dir / "roi_crop_rule_sweep_metrics.csv", rows)
    print("[DONE] crop-rule sweep outputs:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
