#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one-class YOLO lesion detectors against VOC boxes.

Outputs:
  - per-image predicted-box IoU CSV
  - per-fold and aggregate box quality summaries
  - worst-case overlay PNGs
  - optional VOC roots whose Annotations are replaced by detector predictions

The generated predicted-box VOC roots preserve the original JPEGImages,
ImageSets, and manifests layout, so the existing ROI classifiers can read them
without code changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR_SUMMARY = PROJECT_ROOT / "busi_aul_roi_detector_logs" / "busi_aul_yolo_5fold_20260503_181547" / "detector_5fold_summary.csv"
DATA_ROOTS = {
    "busi": PROJECT_ROOT / "busi" / "busi_voc_v3_square_consistent",
    "aul": PROJECT_ROOT / "aul" / "aul_voc_roi_v1",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, min(self.xmin, width - 1.0)),
            max(0.0, min(self.ymin, height - 1.0)),
            max(0.0, min(self.xmax, width - 1.0)),
            max(0.0, min(self.ymax, height - 1.0)),
        )

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(round(self.xmin)), int(round(self.ymin)), int(round(self.xmax)), int(round(self.ymax)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
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


def read_split(root: Path, split: str) -> list[str]:
    split_path = root / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip().split()[0] for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image(root: Path, image_id: str, xml_root: ET.Element | None = None) -> Path:
    image_dir = root / "JPEGImages"
    candidates: list[Path] = []
    filename = xml_root.findtext("filename") if xml_root is not None else None
    if filename:
        candidates.append(image_dir / filename)
    raw = Path(image_id)
    if raw.suffix:
        candidates.append(image_dir / raw.name)
    else:
        candidates.extend(image_dir / f"{image_id}{ext}" for ext in IMAGE_EXTS)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for {image_id} under {image_dir}")


def parse_voc(xml_path: Path) -> tuple[str, int, int, list[Box], ET.Element]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=xml_path.with_suffix(".jpg").name)
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    boxes: list[Box] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = Box(
            float(bnd.findtext("xmin", "0")),
            float(bnd.findtext("ymin", "0")),
            float(bnd.findtext("xmax", "0")),
            float(bnd.findtext("ymax", "0")),
        ).clipped(width, height)
        if box.area > 0:
            boxes.append(box)
    return filename, width, height, boxes, root


def iou(a: Box | None, b: Box | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = Box(max(a.xmin, b.xmin), max(a.ymin, b.ymin), min(a.xmax, b.xmax), min(a.ymax, b.ymax)).area
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_voc_xml(filename: str, width: int, height: int, box: Box | None, label_name: str = "lesion") -> ET.Element:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    if box is not None:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = box.as_int_tuple()
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)
    return root


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def write_voc_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def prepare_pred_voc_root(src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for sub in ["ImageSets", "manifests"]:
        src = src_root / sub
        dst = dst_root / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    (dst_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (dst_root / "Annotations").mkdir(parents=True, exist_ok=True)


def draw_overlay(image_path: Path, gt: Box | None, pred: Box | None, conf: float, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    if gt is not None:
        draw.rectangle(gt.as_int_tuple(), outline=(34, 197, 94), width=4)
    if pred is not None:
        draw.rectangle(pred.as_int_tuple(), outline=(239, 68, 68), width=4)
    draw.text((8, 8), f"green=GT red=Pred conf={conf:.3f} IoU={iou(gt, pred):.3f}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def summarize_rows(rows: list[dict], dataset: str, fold: int, split: str) -> dict:
    vals = [float(r["iou"]) for r in rows]
    confs = [float(r["confidence"]) for r in rows if r["confidence"] != ""]
    n = len(rows)
    no_det = sum(int(r["no_detection"]) for r in rows)
    return {
        "dataset": dataset,
        "fold": fold,
        "split": split,
        "n": n,
        "no_detection": no_det,
        "no_detection_rate": no_det / max(n, 1),
        "mean_iou": mean(vals) if vals else 0.0,
        "median_iou": float(np.median(vals)) if vals else 0.0,
        "recall_iou_0_30": sum(v >= 0.30 for v in vals) / max(n, 1),
        "recall_iou_0_50": sum(v >= 0.50 for v in vals) / max(n, 1),
        "recall_iou_0_75": sum(v >= 0.75 for v in vals) / max(n, 1),
        "mean_confidence": mean(confs) if confs else 0.0,
    }


def aggregate_fold_summaries(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        ds = [r for r in rows if r["dataset"] == dataset and r["split"] == "test"]
        for metric in ["mean_iou", "median_iou", "recall_iou_0_50", "recall_iou_0_75", "no_detection_rate", "mean_confidence"]:
            vals = [float(r[metric]) for r in ds]
            if not vals:
                continue
            s = stdev(vals) if len(vals) > 1 else 0.0
            out.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "metric": metric,
                    "n_folds": len(vals),
                    "mean": mean(vals),
                    "std": s,
                    "mean_std": f"{mean(vals):.4f} +/- {s:.4f}",
                }
            )
    return out


def predict_boxes(model, image_paths: list[Path], imgsz: int, device: str, conf: float) -> list[tuple[Box | None, float]]:
    results = model.predict([str(p) for p in image_paths], imgsz=imgsz, device=device, conf=conf, verbose=False)
    outputs: list[tuple[Box | None, float]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            outputs.append((None, 0.0))
            continue
        confs = boxes.conf.detach().cpu().numpy()
        idx = int(np.argmax(confs))
        xyxy = boxes.xyxy[idx].detach().cpu().numpy().astype(float).tolist()
        outputs.append((Box(*xyxy), float(confs[idx])))
    return outputs


def run_fold(model, dataset: str, fold: int, weights: Path, src_root: Path, out_dir: Path, splits: Iterable[str], imgsz: int, device: str, conf: float, export_pred_voc: bool, max_overlays: int) -> tuple[list[dict], list[dict]]:
    pred_root = out_dir / "pred_voc_roots" / f"{dataset}_fold{fold}_predbox"
    if export_pred_voc:
        if pred_root.exists():
            shutil.rmtree(pred_root)
        prepare_pred_voc_root(src_root, pred_root)

    per_image_rows: list[dict] = []
    fold_summary_rows: list[dict] = []
    for split in splits:
        source_split = f"fold{fold}_val" if split == "val" and not (src_root / "ImageSets" / "Main" / "val.txt").exists() else split
        image_ids = read_split(src_root, source_split)
        image_paths: list[Path] = []
        gt_records: list[tuple[str, str, int, int, Box | None]] = []
        for image_id in image_ids:
            xml_path = src_root / "Annotations" / f"{Path(image_id).stem}.xml"
            filename, width, height, gt_boxes, xml_root = parse_voc(xml_path)
            image_path = find_image(src_root, image_id, xml_root)
            gt = gt_boxes[0] if gt_boxes else None
            image_paths.append(image_path)
            gt_records.append((image_id, filename, width, height, gt))

            if export_pred_voc:
                dst_img = pred_root / "JPEGImages" / image_path.name
                link_or_copy(image_path, dst_img)

        preds = predict_boxes(model, image_paths, imgsz=imgsz, device=device, conf=conf)
        split_rows: list[dict] = []
        for image_path, (image_id, filename, width, height, gt), (pred, pred_conf) in zip(image_paths, gt_records, preds):
            pred = pred.clipped(width, height) if pred is not None else None
            row = {
                "dataset": dataset,
                "fold": fold,
                "split": split,
                "source_split": source_split,
                "image_id": image_id,
                "image_path": str(image_path),
                "gt_xmin": "" if gt is None else gt.xmin,
                "gt_ymin": "" if gt is None else gt.ymin,
                "gt_xmax": "" if gt is None else gt.xmax,
                "gt_ymax": "" if gt is None else gt.ymax,
                "pred_xmin": "" if pred is None else pred.xmin,
                "pred_ymin": "" if pred is None else pred.ymin,
                "pred_xmax": "" if pred is None else pred.xmax,
                "pred_ymax": "" if pred is None else pred.ymax,
                "confidence": "" if pred is None else pred_conf,
                "iou": iou(gt, pred),
                "no_detection": int(pred is None),
                "weights": str(weights),
                "pred_voc_root": str(pred_root) if export_pred_voc else "",
            }
            split_rows.append(row)
            per_image_rows.append(row)

            if export_pred_voc:
                xml = make_voc_xml(filename=filename, width=width, height=height, box=pred)
                write_voc_xml(pred_root / "Annotations" / f"{Path(image_id).stem}.xml", xml)

        fold_summary_rows.append(summarize_rows(split_rows, dataset=dataset, fold=fold, split=split))

        if split == "test" and max_overlays > 0:
            worst = sorted(split_rows, key=lambda r: (float(r["iou"]), -int(r["no_detection"])))[:max_overlays]
            for rank, row in enumerate(worst, 1):
                gt = None if row["gt_xmin"] == "" else Box(float(row["gt_xmin"]), float(row["gt_ymin"]), float(row["gt_xmax"]), float(row["gt_ymax"]))
                pred = None if row["pred_xmin"] == "" else Box(float(row["pred_xmin"]), float(row["pred_ymin"]), float(row["pred_xmax"]), float(row["pred_ymax"]))
                draw_overlay(
                    Path(row["image_path"]),
                    gt,
                    pred,
                    0.0 if row["confidence"] == "" else float(row["confidence"]),
                    out_dir / "overlays" / dataset / f"fold{fold}" / f"worst_{rank:02d}_{Path(str(row['image_id'])).stem}.png",
                )

    return per_image_rows, fold_summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-summary", type=Path, default=DEFAULT_DETECTOR_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_reports" / "busi_aul_detector_box_quality_auto")
    parser.add_argument("--datasets", nargs="+", default=["busi", "aul"], choices=sorted(DATA_ROOTS))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-overlays", type=int, default=12)
    parser.add_argument("--export-pred-voc", type=int, default=1)
    args = parser.parse_args()

    from ultralytics import YOLO

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    if out_dir.name.endswith("_auto"):
        out_dir = out_dir.parent / f"{out_dir.name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    summary_rows = [r for r in read_csv(args.detector_summary) if r.get("dataset") in args.datasets and r.get("return_code") == "0"]
    all_per_image: list[dict] = []
    all_fold_summary: list[dict] = []

    for row in summary_rows:
        dataset = row["dataset"]
        fold = int(row["fold"])
        weights = Path(row["best_weights"])
        src_root = DATA_ROOTS[dataset]
        print(f"[EVAL] dataset={dataset} fold={fold} weights={weights}")
        model = YOLO(str(weights))
        per_image, fold_summary = run_fold(
            model=model,
            dataset=dataset,
            fold=fold,
            weights=weights,
            src_root=src_root,
            out_dir=out_dir,
            splits=args.splits,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            export_pred_voc=bool(args.export_pred_voc),
            max_overlays=args.max_overlays,
        )
        all_per_image.extend(per_image)
        all_fold_summary.extend(fold_summary)
        write_csv(out_dir / "box_quality_per_image.csv", all_per_image)
        write_csv(out_dir / "box_quality_per_fold.csv", all_fold_summary)
        write_csv(out_dir / "box_quality_aggregate.csv", aggregate_fold_summaries(all_fold_summary))

    aggregate = aggregate_fold_summaries(all_fold_summary)
    md = ["# BUSI/AUL detector box-quality summary", ""]
    for dataset in args.datasets:
        md.append(f"## {dataset.upper()}")
        for row in aggregate:
            if row["dataset"] == dataset:
                md.append(f"- {row['metric']}: {row['mean_std']}")
        md.append("")
    md.extend(
        [
            "## Outputs",
            f"- per-image CSV: `{out_dir / 'box_quality_per_image.csv'}`",
            f"- per-fold CSV: `{out_dir / 'box_quality_per_fold.csv'}`",
            f"- aggregate CSV: `{out_dir / 'box_quality_aggregate.csv'}`",
            f"- overlays: `{out_dir / 'overlays'}`",
            f"- predicted VOC roots: `{out_dir / 'pred_voc_roots'}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
