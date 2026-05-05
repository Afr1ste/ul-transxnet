#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export YOLO detector boxes for TN5000 splits without running classification."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import eval_tn5000_auto_roi_pipeline as autoeval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tn5000-root", default=r"C:\Users\Afr1ste\PycharmProjects\Thyroid\TN5000_forReview")
    p.add_argument("--detector-weights", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--det-imgsz", type=int, default=640)
    p.add_argument("--det-conf", type=float, default=0.001)
    p.add_argument("--det-iou", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.tn5000_root)
    detector_weights = Path(args.detector_weights)
    if not detector_weights.exists():
        raise FileNotFoundError(detector_weights)
    out_dir = Path(args.output_dir) if args.output_dir else Path("eval_reports") / ("tn5000_detector_boxes_" + time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_device = "0" if args.device == "cuda" else args.device
    samples_by_split = {split: autoeval.load_samples(root, split) for split in args.splits}
    autoeval.run_detector(
        detector_weights=detector_weights,
        samples_by_split=samples_by_split,
        output_dir=out_dir,
        imgsz=args.det_imgsz,
        conf=args.det_conf,
        iou=args.det_iou,
        device=yolo_device,
    )
    print("[DONE] detector boxes exported:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
