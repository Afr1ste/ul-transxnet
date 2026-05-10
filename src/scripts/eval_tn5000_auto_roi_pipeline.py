#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate automatic ROI detection followed by the existing TN5000 classifier.

This script keeps the lesion detector diagnosis-agnostic: the YOLO model predicts
one class only ("lesion"). The predicted box is then expanded with the same
TN5000 ROI crop rule used by the current classifier before classification.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

import fl_tn5000_roi_compare_multimodel as trainmod


VALID_IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tn5000-root", default=r"<LOCAL_THYROID_ROOT>\TN5000_forReview")
    p.add_argument("--detector-weights", required=True)
    p.add_argument("--classifier-log-dir", default=r"tn5000_ggg_mca_enabled_3seed_logs\20260426_093728")
    p.add_argument("--output-dir", default="")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--modes", nargs="+", default=["oracle", "auto", "full"], choices=["oracle", "auto", "full"])
    p.add_argument("--det-imgsz", type=int, default=640)
    p.add_argument("--det-conf", type=float, default=0.001)
    p.add_argument("--det-iou", type=float, default=0.7)
    p.add_argument("--classifier-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--bbox-expand-ratio", type=float, default=0.30)
    p.add_argument("--min-crop-size", type=int, default=64)
    p.add_argument("--square-crop", type=int, default=0, help="0 matches current TN5000 classifier code; 1 square-pads crop around box center.")
    p.add_argument("--hflip-tta", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def read_split_ids(root: Path, split: str) -> List[str]:
    split_file = root / "ImageSets" / "Main" / f"{split}.txt"
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_image_path(root: Path, image_id: str) -> Path:
    image_dir = root / "JPEGImages"
    for suffix in VALID_IMAGE_SUFFIXES:
        p = image_dir / f"{image_id}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for id={image_id}")


def parse_xml_label_bbox(xml_path: Path) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels: List[int] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name in {"0", "benign"}:
            labels.append(0)
        elif name in {"1", "malignant"}:
            labels.append(1)
        else:
            raise ValueError(f"Unknown label {name!r} in {xml_path}")
        bnd = obj.find("bndbox")
        if bnd is not None:
            x1 = float(bnd.findtext("xmin", default="0"))
            y1 = float(bnd.findtext("ymin", default="0"))
            x2 = float(bnd.findtext("xmax", default="0"))
            y2 = float(bnd.findtext("ymax", default="0"))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    if not labels:
        raise ValueError(f"No labels in {xml_path}")
    label = max(labels)
    if not boxes:
        return label, None
    return label, (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def load_samples(root: Path, split: str) -> List[Dict]:
    samples: List[Dict] = []
    for image_id in read_split_ids(root, split):
        image_path = find_image_path(root, image_id)
        xml_path = root / "Annotations" / f"{image_id}.xml"
        label, bbox = parse_xml_label_bbox(xml_path)
        samples.append(
            {
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path),
                "xml_path": str(xml_path),
                "label": int(label),
                "gt_bbox": bbox,
            }
        )
    return samples


def box_iou(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def run_detector(
    detector_weights: Path,
    samples_by_split: Dict[str, List[Dict]],
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> Dict[str, Dict[str, Dict]]:
    detector = YOLO(str(detector_weights))
    predictions: Dict[str, Dict[str, Dict]] = {}
    detect_dir = output_dir / "detector_predictions"
    detect_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in samples_by_split.items():
        split_preds: Dict[str, Dict] = {}
        rows: List[Dict] = []
        for sample in tqdm(samples, desc=f"Detect-{split}"):
            result = detector.predict(
                source=sample["image_path"],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            pred_bbox = None
            pred_conf = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                best_idx = int(np.argmax(scores))
                pred_conf = float(scores[best_idx])
                pred_bbox = tuple(float(x) for x in xyxy[best_idx].tolist())
            iou_gt = box_iou(pred_bbox, sample["gt_bbox"])
            pred = {
                "pred_bbox": pred_bbox,
                "pred_conf": pred_conf,
                "iou_gt": iou_gt,
                "no_detection": pred_bbox is None,
            }
            split_preds[sample["image_id"]] = pred
            rows.append(
                {
                    "split": split,
                    "image_id": sample["image_id"],
                    "image_path": sample["image_path"],
                    "label": sample["label"],
                    "gt_bbox": box_to_str(sample["gt_bbox"]),
                    "pred_bbox": box_to_str(pred_bbox),
                    "pred_conf": pred_conf,
                    "iou_gt": iou_gt,
                    "no_detection": int(pred_bbox is None),
                }
            )
        predictions[split] = split_preds
        write_csv(detect_dir / f"{split}_boxes.csv", rows)
        write_json(detect_dir / f"{split}_box_metrics.json", summarize_detector_rows(rows))
    return predictions


def summarize_detector_rows(rows: Sequence[Dict]) -> Dict:
    ious = np.array([float(r["iou_gt"]) for r in rows], dtype=np.float32)
    no_det = np.array([int(r["no_detection"]) for r in rows], dtype=np.int64)
    return {
        "n": int(len(rows)),
        "no_detection": int(no_det.sum()),
        "no_detection_rate": float(no_det.mean()) if len(no_det) else math.nan,
        "mean_iou": float(ious.mean()) if len(ious) else math.nan,
        "median_iou": float(np.median(ious)) if len(ious) else math.nan,
        "recall_iou_0_30": float((ious >= 0.30).mean()) if len(ious) else math.nan,
        "recall_iou_0_50": float((ious >= 0.50).mean()) if len(ious) else math.nan,
        "recall_iou_0_75": float((ious >= 0.75).mean()) if len(ious) else math.nan,
    }


def parse_box_string(s: str) -> Optional[Tuple[float, float, float, float]]:
    s = str(s or "").strip()
    if not s:
        return None
    parts = [float(x) for x in s.split()]
    if len(parts) != 4:
        raise ValueError(f"Invalid box string: {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def box_to_str(box: Optional[Tuple[float, float, float, float]]) -> str:
    if box is None:
        return ""
    return "%.3f %.3f %.3f %.3f" % (box[0], box[1], box[2], box[3])


def expand_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    expand_ratio: float,
    min_crop_size: int,
    square_crop: bool,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    bw = max(float(x2 - x1), float(min_crop_size))
    bh = max(float(y2 - y1), float(min_crop_size))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = bw * (1.0 + 2.0 * expand_ratio)
    new_h = bh * (1.0 + 2.0 * expand_ratio)
    if square_crop:
        side = max(new_w, new_h)
        new_w = side
        new_h = side
    nx1 = int(round(cx - new_w / 2.0))
    ny1 = int(round(cy - new_h / 2.0))
    nx2 = int(round(cx + new_w / 2.0))
    ny2 = int(round(cy + new_h / 2.0))
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(width, nx2), min(height, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


class AutoRoiClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict],
        detector_preds: Dict[str, Dict],
        mode: str,
        transform,
        expand_ratio: float,
        min_crop_size: int,
        square_crop: bool,
    ):
        self.samples = list(samples)
        self.detector_preds = detector_preds
        self.mode = mode
        self.transform = transform
        self.expand_ratio = expand_ratio
        self.min_crop_size = min_crop_size
        self.square_crop = square_crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        crop_source = "full"
        crop_bbox = None
        if self.mode == "oracle":
            crop_bbox = sample["gt_bbox"]
            crop_source = "oracle"
        elif self.mode == "auto":
            pred = self.detector_preds.get(sample["image_id"], {})
            crop_bbox = pred.get("pred_bbox")
            crop_source = "auto" if crop_bbox is not None else "full_fallback"
        elif self.mode == "full":
            crop_bbox = None
            crop_source = "full"
        else:
            raise ValueError(f"Unsupported crop mode: {self.mode}")

        if crop_bbox is not None:
            expanded = expand_box(
                crop_bbox,
                img.width,
                img.height,
                self.expand_ratio,
                self.min_crop_size,
                self.square_crop,
            )
            if expanded is not None:
                img = img.crop(expanded)
            else:
                crop_source = "full_fallback"
        if self.transform is not None:
            img = self.transform(img)
        return img, int(sample["label"]), sample["image_id"], sample["image_path"], crop_source


def build_eval_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def configure_classifier(input_size: int):
    trainmod.Config.model_family = "custom"
    trainmod.Config.backbone_name = "transxnet_t"
    trainmod.Config.backbone_module = "models.transxnetggg"
    trainmod.Config.backbone_func = "transxnet_t"
    trainmod.Config.backbone_out_dim = 1000
    trainmod.Config.input_size = int(input_size)
    trainmod.Config.dropout = 0.30


def parse_ckpts_from_summary(summary_path: Path) -> List[Path]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"ensemble_ckpts_used:\s*(\[.*?\])", text, flags=re.S)
    if not m:
        raise ValueError(f"Could not parse ensemble_ckpts_used from {summary_path}")
    raw = ast.literal_eval(m.group(1))
    out: List[Path] = []
    for item in raw:
        p = Path(str(item))
        if not p.is_absolute():
            candidates = [
                Path.cwd() / p,
                summary_path.parent / p.name,
                summary_path.parent.parent.parent / p,
            ]
            p = next((c for c in candidates if c.exists()), candidates[0])
        out.append(p)
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + "; ".join(missing))
    return out


def load_classifier_ckpts(classifier_log_dir: Path) -> List[Path]:
    all_runs = classifier_log_dir / "all_runs_metrics.csv"
    if not all_runs.exists():
        raise FileNotFoundError(f"Missing all_runs_metrics.csv: {all_runs}")
    ckpts: List[Path] = []
    with all_runs.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run_dir = Path(row["run_dir"])
            ckpts.extend(parse_ckpts_from_summary(run_dir / "summary.txt"))
    seen = set()
    unique = []
    for p in ckpts:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


@torch.no_grad()
def collect_logits_for_ckpt(model, ckpt: Path, loader: DataLoader, device: torch.device, hflip_tta: bool, desc: str):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logits_list, labels_list = [], []
    ids, paths, sources = [], [], []
    for inputs, labels, image_ids, image_paths, crop_sources in tqdm(loader, desc=desc):
        inputs = inputs.to(device)
        logits = model(inputs)
        if hflip_tta:
            logits_flip = model(torch.flip(inputs, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())
        ids.extend(list(image_ids))
        paths.extend(list(image_paths))
        sources.extend(list(crop_sources))
    return (
        torch.cat(labels_list).numpy(),
        torch.cat(logits_list).numpy(),
        ids,
        paths,
        sources,
    )


def collect_ensemble_logits(
    ckpts: Sequence[Path],
    loader: DataLoader,
    device: torch.device,
    hflip_tta: bool,
    split: str,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    model = trainmod.UnifiedRoiClassifier(num_classes=2).to(device)
    logits_sum = None
    labels_ref = None
    ids_ref = None
    paths_ref = None
    sources_ref = None
    for idx, ckpt in enumerate(ckpts, start=1):
        labels, logits, ids, paths, sources = collect_logits_for_ckpt(
            model,
            ckpt,
            loader,
            device,
            hflip_tta,
            desc=f"Cls-{split}-{mode}-{idx:02d}/{len(ckpts):02d}",
        )
        if labels_ref is None:
            labels_ref, ids_ref, paths_ref, sources_ref = labels, ids, paths, sources
        else:
            if not np.array_equal(labels_ref, labels) or ids_ref != ids:
                raise RuntimeError("Classifier dataloader order changed across checkpoints")
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return labels_ref, logits_sum / float(len(ckpts)), ids_ref, paths_ref, sources_ref


def logits_to_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = torch.tensor(logits / max(float(temperature), 1e-6), dtype=torch.float32)
    return torch.softmax(x, dim=1).numpy()[:, 1]


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    return trainmod.fit_temperature_on_val(logits, labels, device)


def scan_thresholds(labels: np.ndarray, probs: np.ndarray, start: float = 0.10, end: float = 0.95, step: float = 0.01):
    rows = []
    n = int(round((end - start) / step)) + 1
    for i in range(n):
        thr = round(start + i * step, 4)
        rows.append(compute_metrics(labels, probs, thr))
    rows = sorted(rows, key=lambda r: (r["bal_acc"], r["f1_macro"], -abs(r["threshold"] - 0.5)), reverse=True)
    return rows


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = math.nan
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": auc,
        "recall_0": float(recall_score(labels, preds, labels=[0, 1], average=None, zero_division=0)[0]),
        "recall_1": float(recall_score(labels, preds, labels=[0, 1], average=None, zero_division=0)[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_predictions(
    path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    ids: Sequence[str],
    paths: Sequence[str],
    sources: Sequence[str],
    threshold: float,
):
    rows = []
    preds = (probs >= threshold).astype(int)
    for image_id, image_path, y, p, prob, source in zip(ids, paths, labels, preds, probs, sources):
        rows.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "true_label": int(y),
                "pred_label": int(p),
                "prob_class1": float(prob),
                "threshold": float(threshold),
                "crop_source": source,
                "is_wrong": int(int(y) != int(p)),
            }
        )
    write_csv(path, rows)


def write_csv(path: Path, rows: Sequence[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.tn5000_root)
    detector_weights = Path(args.detector_weights)
    if not detector_weights.exists():
        raise FileNotFoundError(detector_weights)
    out_dir = Path(args.output_dir) if args.output_dir else Path("eval_reports") / ("tn5000_auto_roi_pipeline_" + time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    yolo_device = "0" if device.type == "cuda" else "cpu"
    configure_classifier(args.input_size)

    samples_by_split = {split: load_samples(root, split) for split in args.splits}
    detector_preds = run_detector(
        detector_weights,
        samples_by_split,
        out_dir,
        imgsz=args.det_imgsz,
        conf=args.det_conf,
        iou=args.det_iou,
        device=yolo_device,
    )
    ckpts = load_classifier_ckpts(Path(args.classifier_log_dir))
    write_json(out_dir / "classifier_ckpts_used.json", [str(p) for p in ckpts])

    transform = build_eval_transform(args.input_size)
    summary_rows: List[Dict] = []
    for mode in args.modes:
        logits_by_split = {}
        labels_by_split = {}
        ids_by_split = {}
        paths_by_split = {}
        sources_by_split = {}
        for split in args.splits:
            ds = AutoRoiClassificationDataset(
                samples_by_split[split],
                detector_preds.get(split, {}),
                mode=mode,
                transform=transform,
                expand_ratio=args.bbox_expand_ratio,
                min_crop_size=args.min_crop_size,
                square_crop=bool(args.square_crop),
            )
            loader = DataLoader(ds, batch_size=args.classifier_batch_size, shuffle=False, num_workers=args.num_workers)
            labels, logits, ids, paths, sources = collect_ensemble_logits(
                ckpts,
                loader,
                device=device,
                hflip_tta=bool(args.hflip_tta),
                split=split,
                mode=mode,
            )
            logits_by_split[split] = logits
            labels_by_split[split] = labels
            ids_by_split[split] = ids
            paths_by_split[split] = paths
            sources_by_split[split] = sources

        if "val" not in logits_by_split:
            raise RuntimeError("The val split is required for temperature and threshold selection.")
        temperature = fit_temperature(logits_by_split["val"], labels_by_split["val"], device)
        val_probs = logits_to_probs(logits_by_split["val"], temperature)
        threshold_rows = scan_thresholds(labels_by_split["val"], val_probs)
        best_thr = float(threshold_rows[0]["threshold"])
        mode_dir = out_dir / f"classifier_{mode}"
        write_csv(mode_dir / "val_threshold_scan.csv", threshold_rows)

        for split in args.splits:
            probs = logits_to_probs(logits_by_split[split], temperature)
            metrics = compute_metrics(labels_by_split[split], probs, best_thr)
            metrics.update({"mode": mode, "split": split, "temperature": float(temperature)})
            summary_rows.append(metrics)
            write_predictions(
                mode_dir / f"{split}_predictions.csv",
                labels_by_split[split],
                probs,
                ids_by_split[split],
                paths_by_split[split],
                sources_by_split[split],
                best_thr,
            )

    write_csv(out_dir / "auto_roi_classification_metrics.csv", summary_rows)
    write_json(
        out_dir / "run_config.json",
        {
            "tn5000_root": str(root),
            "detector_weights": str(detector_weights),
            "classifier_log_dir": str(Path(args.classifier_log_dir)),
            "splits": args.splits,
            "modes": args.modes,
            "det_imgsz": args.det_imgsz,
            "det_conf": args.det_conf,
            "det_iou": args.det_iou,
            "input_size": args.input_size,
            "bbox_expand_ratio": args.bbox_expand_ratio,
            "square_crop": bool(args.square_crop),
            "hflip_tta": bool(args.hflip_tta),
            "device": str(device),
        },
    )
    print("[DONE] auto ROI pipeline outputs:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
