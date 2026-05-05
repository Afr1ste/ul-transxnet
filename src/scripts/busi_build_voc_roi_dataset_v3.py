from __future__ import annotations

"""
BUSI lesion-only -> VOC ROI dataset builder (v3 square-consistent)

用途
----
把 BUSI 原始目录:
    C:/Users/Afr1ste/PycharmProjects/Thyroid/Dataset_BUSI_with_GT
下的 benign / malignant 图像与 mask，转换成与你当前 TN3K 训练脚本兼容的
VOC 风格 ROI 分类数据集。

输出目录结构
------------
OUTPUT_ROOT/
├─ JPEGImages
├─ Annotations
├─ ImageSets/Main
│  ├─ trainval.txt
│  ├─ test.txt
│  ├─ fold0_train.txt
│  ├─ fold0_val.txt
│  └─ ...
├─ manifests
│  ├─ label_manifest.csv
│  ├─ bbox_manifest.csv
│  ├─ original_records.csv
│  ├─ split_assignments.json
│  ├─ missing_or_skipped.csv
│  └─ split_bbox_stats.csv
└─ bbox_previews

与当前训练脚本的对齐点
----------------------
1) 训练脚本读取:
   - ImageSets/Main/{split}.txt
   - manifests/label_manifest.csv
   - Annotations/{image_id}.xml
   - JPEGImages/{image_id}.png/.jpg

2) label_manifest.csv 至少会写出:
   - new_filename
   - label
   - split
   这正是你当前 VOCROIDataset 所要求的关键列。

3) XML 中只写单个 object/bndbox，训练脚本会把它当 ROI 框来 crop。

运行方式
--------
直接在 PyCharm 里运行本文件即可。
你只需要修改下面【用户配置区】里的路径和少量开关。

当前默认策略
------------
- 仅使用 benign / malignant 两类
- normal 默认不参与这个 ROI 二分类构建
- 同一原图的多个 mask 会先合并，再做连通域筛选和 ROI 生成
- split 为 image-level stratified split:
    test_ratio = 0.20
    trainval 再做 5-fold stratified
- ROI 逻辑采用你 TN3K 的 v3 square-consistent 风格：
    主连通域 + 邻近有效区域 -> raw union bbox -> centered square crop

后续如何训练
------------
构建完成后，把训练脚本里的:
    Config.data_root
改成这个 OUTPUT_ROOT，
并保持:
    train_split = "fold0_train"
    val_split   = "fold0_val"
    test_split  = "test"
即可直接复用你现有 TN3K ROI 分类训练脚本。

注意
----
1) 这是 lesion-only 二分类构建脚本，不处理 normal 类的整图分类逻辑。
2) split 是 image-level，不是 patient-level。
3) 为了避免图像重编码损失，默认保留原图扩展名（通常是 .png），不强制改成 .jpg。
"""

import csv
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# =============================================================================
# 用户配置区：直接改这里
# =============================================================================
SOURCE_ROOT = Path(os.environ.get("BUSI_SOURCE_ROOT", r"C:/Users/Afr1ste/PycharmProjects/Thyroid/Dataset_BUSI_with_GT"))
OUTPUT_ROOT = Path(os.environ.get("BUSI_OUTPUT_ROOT", r"C:\Users\Afr1ste\PycharmProjects\Thyroid\busi\busi_voc_v3_square_consistent"))

INCLUDE_CLASSES = ["benign", "malignant"]
CLASS_TO_LABEL = {
    "benign": 0,
    "malignant": 1,
}

TEST_RATIO = 0.20
N_FOLDS = 5
SPLIT_SEED = 42

OVERWRITE_OUTPUT_ROOT = True
COPY_IMAGES = True
MAKE_PREVIEWS = True
MAX_PREVIEWS_PER_SPLIT = 120

# =============================================================================
# ROI / bbox 规则（BUSI 复用 TN3K v3 square-consistent）
# =============================================================================
MIN_COMPONENT_AREA = 80
MIN_RELATIVE_AREA = 0.10
NEAR_MARGIN_RATIO = 0.12
CENTROID_DIST_RATIO = 0.45
XML_OBJECT_NAME = "0"

MIN_FINAL_SIDE_RATIO = 0.30
MAX_FINAL_SIDE_RATIO = 0.68

SMALL_BOX_AREA_RATIO_1 = 0.03
SMALL_BOX_AREA_RATIO_2 = 0.10
MEDIUM_BOX_AREA_RATIO = 0.25

SCALE_TINY = 1.80
SCALE_SMALL = 1.55
SCALE_MEDIUM = 1.35
SCALE_LARGE = 1.18

TOO_SMALL_AREA_RATIO = 0.015
TOO_LARGE_AREA_RATIO = 0.85
NEAR_FULL_IMAGE_AREA_RATIO = 0.95
EDGE_NEAR_RATIO = 0.02

# =============================================================================
# 输出固定目录
# =============================================================================
JPEG_DIR = OUTPUT_ROOT / "JPEGImages"
ANN_DIR = OUTPUT_ROOT / "Annotations"
IMAGESETS_MAIN_DIR = OUTPUT_ROOT / "ImageSets" / "Main"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
PREVIEW_DIR = OUTPUT_ROOT / "bbox_previews"

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def reset_output_root() -> None:
    if OVERWRITE_OUTPUT_ROOT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for d in [OUTPUT_ROOT, JPEG_DIR, ANN_DIR, IMAGESETS_MAIN_DIR, MANIFEST_DIR, PREVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sample"


def extract_case_token(stem: str) -> str:
    m = re.search(r"\((\d+)\)\s*$", stem)
    if m:
        return f"{int(m.group(1)):04d}"
    return sanitize_name(stem)


def build_new_stem(split_name: str, class_name: str, orig_stem: str) -> str:
    case_token = extract_case_token(orig_stem)
    class_token = sanitize_name(class_name)
    return f"{split_name}_{class_token}_{case_token}"


def clip_box(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> Tuple[int, int, int, int]:
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width - 1))
    ymax = max(0, min(ymax, height - 1))
    if xmax <= xmin:
        xmax = min(width - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(height - 1, ymin + 1)
    return xmin, ymin, xmax, ymax


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def expand_box(box: Tuple[int, int, int, int], ratio: float, width: int, height: int) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    ex = int(round(bw * ratio))
    ey = int(round(bh * ratio))
    return clip_box(xmin - ex, ymin - ey, xmax + ex, ymax + ey, width, height)


def choose_square_scale(raw_area_ratio: float) -> float:
    if raw_area_ratio < SMALL_BOX_AREA_RATIO_1:
        return SCALE_TINY
    if raw_area_ratio < SMALL_BOX_AREA_RATIO_2:
        return SCALE_SMALL
    if raw_area_ratio < MEDIUM_BOX_AREA_RATIO:
        return SCALE_MEDIUM
    return SCALE_LARGE


def make_centered_square_crop(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    raw_area_ratio: float,
) -> Tuple[Tuple[int, int, int, int], Dict[str, float]]:
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    raw_square_side = max(bw, bh)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    short_edge = min(width, height)
    min_side = max(2, int(round(short_edge * MIN_FINAL_SIDE_RATIO)))
    max_side = max(min_side + 1, int(round(short_edge * MAX_FINAL_SIDE_RATIO)))

    square_scale = choose_square_scale(raw_area_ratio)
    target_side = int(round(raw_square_side * square_scale))
    final_side = int(np.clip(target_side, min_side, max_side))

    half = (final_side - 1) / 2.0
    sq_xmin = int(round(cx - half))
    sq_ymin = int(round(cy - half))
    sq_xmax = sq_xmin + final_side - 1
    sq_ymax = sq_ymin + final_side - 1

    if sq_xmin < 0:
        shift = -sq_xmin
        sq_xmin += shift
        sq_xmax += shift
    if sq_ymin < 0:
        shift = -sq_ymin
        sq_ymin += shift
        sq_ymax += shift
    if sq_xmax > width - 1:
        shift = sq_xmax - (width - 1)
        sq_xmin -= shift
        sq_xmax -= shift
    if sq_ymax > height - 1:
        shift = sq_ymax - (height - 1)
        sq_ymin -= shift
        sq_ymax -= shift

    sq_xmin, sq_ymin, sq_xmax, sq_ymax = clip_box(sq_xmin, sq_ymin, sq_xmax, sq_ymax, width, height)

    final_w = sq_xmax - sq_xmin + 1
    final_h = sq_ymax - sq_ymin + 1
    realized_side = max(final_w, final_h)

    info = {
        "raw_square_side": int(raw_square_side),
        "square_scale": round(square_scale, 4),
        "target_square_side": int(target_side),
        "min_square_side": int(min_side),
        "max_square_side": int(max_side),
        "final_square_side": int(realized_side),
        "final_square_side_ratio": round(realized_side / float(short_edge), 6),
    }
    return (sq_xmin, sq_ymin, sq_xmax, sq_ymax), info


def compute_qc_flags(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Dict[str, object]:
    xmin, ymin, xmax, ymax = bbox
    box_w = xmax - xmin + 1
    box_h = ymax - ymin + 1
    box_area = box_w * box_h
    img_area = width * height
    area_ratio = box_area / float(img_area)

    edge_near_x = max(1, int(round(width * EDGE_NEAR_RATIO)))
    edge_near_y = max(1, int(round(height * EDGE_NEAR_RATIO)))

    touch_left = xmin <= 0
    touch_top = ymin <= 0
    touch_right = xmax >= width - 1
    touch_bottom = ymax >= height - 1

    near_left = xmin <= edge_near_x
    near_top = ymin <= edge_near_y
    near_right = (width - 1 - xmax) <= edge_near_x
    near_bottom = (height - 1 - ymax) <= edge_near_y

    return {
        "box_w": box_w,
        "box_h": box_h,
        "box_area": box_area,
        "box_area_ratio": round(area_ratio, 6),
        "touch_left": int(touch_left),
        "touch_top": int(touch_top),
        "touch_right": int(touch_right),
        "touch_bottom": int(touch_bottom),
        "touch_any": int(touch_left or touch_top or touch_right or touch_bottom),
        "touch_both_lr": int(touch_left and touch_right),
        "touch_both_tb": int(touch_top and touch_bottom),
        "near_left": int(near_left),
        "near_top": int(near_top),
        "near_right": int(near_right),
        "near_bottom": int(near_bottom),
        "near_any_edge": int(near_left or near_top or near_right or near_bottom),
        "too_small": int(area_ratio < TOO_SMALL_AREA_RATIO),
        "too_large": int(area_ratio > TOO_LARGE_AREA_RATIO),
        "near_full_image": int(area_ratio > NEAR_FULL_IMAGE_AREA_RATIO),
    }


def is_mask_stem(stem: str) -> bool:
    return re.search(r"_mask(?:_\d+)?$", stem) is not None


def strip_mask_suffix(stem: str) -> str:
    return re.sub(r"_mask(?:_\d+)?$", "", stem)


def discover_busi_records() -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    skipped: List[Dict] = []

    for class_name in INCLUDE_CLASSES:
        class_dir = SOURCE_ROOT / class_name
        if not class_dir.exists():
            skipped.append({
                "class_name": class_name,
                "reason": "class_dir_missing",
                "path": str(class_dir),
            })
            continue

        all_files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES])
        image_files = [p for p in all_files if not is_mask_stem(p.stem)]
        mask_files = [p for p in all_files if is_mask_stem(p.stem)]

        base_to_masks: Dict[str, List[Path]] = defaultdict(list)
        for mp in mask_files:
            base_to_masks[strip_mask_suffix(mp.stem)].append(mp)

        for img_path in image_files:
            base_stem = img_path.stem
            matched_masks = sorted(base_to_masks.get(base_stem, []))
            if len(matched_masks) == 0:
                skipped.append({
                    "class_name": class_name,
                    "orig_filename": img_path.name,
                    "reason": "mask_not_found",
                    "image_path": str(img_path),
                })
                continue

            records.append({
                "class_name": class_name,
                "label": CLASS_TO_LABEL[class_name],
                "orig_filename": img_path.name,
                "orig_stem": img_path.stem,
                "orig_suffix": img_path.suffix.lower(),
                "image_path": str(img_path),
                "mask_paths": [str(p) for p in matched_masks],
                "n_mask_files": len(matched_masks),
            })

    return records, skipped


def load_merged_mask(mask_paths: List[Path]) -> np.ndarray:
    merged: Optional[np.ndarray] = None
    for mp in mask_paths:
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mp}")
        binary = (mask > 0).astype(np.uint8) * 255
        if merged is None:
            merged = binary
        else:
            if merged.shape != binary.shape:
                raise ValueError(f"Mask shape mismatch: {mp}, shape={binary.shape}, merged={merged.shape}")
            merged = np.maximum(merged, binary)
    if merged is None:
        raise ValueError("No masks provided")
    return merged


def compute_bbox_from_mask(mask: np.ndarray) -> Dict:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        raise ValueError("Empty mask")

    height, width = binary.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise ValueError("No foreground component found")

    comps: List[Dict] = []
    for comp_id in range(1, num_labels):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[comp_id]
        comps.append({
            "id": comp_id,
            "area": area,
            "bbox": (x, y, x + w - 1, y + h - 1),
            "centroid": (float(cx), float(cy)),
        })

    main = max(comps, key=lambda c: c["area"])
    largest_area = main["area"]
    main_bbox = main["bbox"]
    mx1, my1, mx2, my2 = main_bbox
    main_w = mx2 - mx1 + 1
    main_h = my2 - my1 + 1
    main_scale = max(main_w, main_h)
    main_cx, main_cy = main["centroid"]

    expanded_main_bbox = expand_box(main_bbox, NEAR_MARGIN_RATIO, width, height)
    area_threshold = max(MIN_COMPONENT_AREA, int(round(largest_area * MIN_RELATIVE_AREA)))

    kept = [main]
    kept_reason_map = {main["id"]: "main_component"}

    for comp in comps:
        if comp["id"] == main["id"]:
            continue

        area = comp["area"]
        if area < area_threshold:
            continue

        comp_bbox = comp["bbox"]
        comp_cx, comp_cy = comp["centroid"]
        centroid_dist = float(np.hypot(comp_cx - main_cx, comp_cy - main_cy))
        near_by_centroid = centroid_dist <= (CENTROID_DIST_RATIO * main_scale)
        near_by_bbox = boxes_intersect(comp_bbox, expanded_main_bbox)

        if near_by_centroid or near_by_bbox:
            kept.append(comp)
            if near_by_bbox and near_by_centroid:
                kept_reason_map[comp["id"]] = "near_bbox+centroid"
            elif near_by_bbox:
                kept_reason_map[comp["id"]] = "near_bbox"
            else:
                kept_reason_map[comp["id"]] = "near_centroid"

    xmin = min(c["bbox"][0] for c in kept)
    ymin = min(c["bbox"][1] for c in kept)
    xmax = max(c["bbox"][2] for c in kept)
    ymax = max(c["bbox"][3] for c in kept)

    raw_union_bbox = (xmin, ymin, xmax, ymax)
    raw_w = xmax - xmin + 1
    raw_h = ymax - ymin + 1
    raw_area_ratio = (raw_w * raw_h) / float(width * height)

    final_bbox, square_info = make_centered_square_crop(raw_union_bbox, width, height, raw_area_ratio)

    return {
        "bbox": final_bbox,
        "raw_union_bbox": raw_union_bbox,
        "main_bbox": main_bbox,
        "largest_area": largest_area,
        "all_components": len(comps),
        "kept_components": len(kept),
        "kept_component_ids": ",".join(str(c["id"]) for c in kept),
        "kept_reasons": "|".join(f"{k}:{v}" for k, v in sorted(kept_reason_map.items())),
        "area_threshold": area_threshold,
        "raw_area_ratio": round(raw_area_ratio, 6),
        "crop_mode": "square_consistent_v3",
        **square_info,
    }


def write_voc_xml(
    xml_path: Path,
    image_filename: str,
    image_abs_path: Path,
    width: int,
    height: int,
    depth: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> None:
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2007"

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename

    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = str(image_abs_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "BUSI"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = XML_OBJECT_NAME
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=False)


def save_preview(
    image_path: Path,
    preview_path: Path,
    final_bbox: Tuple[int, int, int, int],
    raw_bbox: Tuple[int, int, int, int],
    main_bbox: Tuple[int, int, int, int],
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(raw_bbox, outline=(255, 215, 0), width=2)
    draw.rectangle(main_bbox, outline=(0, 255, 0), width=2)
    draw.rectangle(final_bbox, outline=(255, 0, 0), width=3)
    img.save(preview_path)


def build_stratified_trainval_test(records: List[Dict]) -> Tuple[List[int], List[int]]:
    rng = random.Random(SPLIT_SEED)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        label_to_indices[int(r["label"])].append(idx)

    trainval_indices: List[int] = []
    test_indices: List[int] = []

    for label, idxs in sorted(label_to_indices.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)

        if len(idxs) == 1:
            n_test = 0
        else:
            n_test = int(round(len(idxs) * TEST_RATIO))
            n_test = max(1, min(n_test, len(idxs) - 1))

        test_part = idxs[:n_test]
        trainval_part = idxs[n_test:]

        test_indices.extend(test_part)
        trainval_indices.extend(trainval_part)

    return sorted(trainval_indices), sorted(test_indices)


def build_stratified_folds(trainval_indices: List[int], records: List[Dict], n_folds: int) -> List[Dict[str, List[int]]]:
    rng = random.Random(SPLIT_SEED + 1000)
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in trainval_indices:
        label_to_indices[int(records[idx]["label"])].append(idx)

    fold_buckets: List[List[int]] = [[] for _ in range(n_folds)]
    for label, idxs in sorted(label_to_indices.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_buckets[i % n_folds].append(idx)

    folds: List[Dict[str, List[int]]] = []
    all_trainval_set = set(trainval_indices)
    for fold_id in range(n_folds):
        val_set = set(sorted(fold_buckets[fold_id]))
        train_set = sorted(all_trainval_set - val_set)
        val_list = sorted(val_set)
        folds.append({"train": train_set, "val": val_list})
    return folds


def process_records(records: List[Dict], split_assignments: Dict[int, str]) -> Tuple[List[Dict], List[Dict]]:
    ok_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    preview_counts: Dict[str, int] = defaultdict(int)

    for idx, record in enumerate(records):
        split_name = split_assignments[idx]
        image_path = Path(record["image_path"])
        mask_paths = [Path(p) for p in record["mask_paths"]]
        class_name = str(record["class_name"])
        label = int(record["label"])

        if not image_path.exists():
            skipped_rows.append({
                "split": split_name,
                "class_name": class_name,
                "orig_filename": record["orig_filename"],
                "reason": "image_missing",
            })
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                depth = len(img.getbands())
        except Exception as e:
            skipped_rows.append({
                "split": split_name,
                "class_name": class_name,
                "orig_filename": record["orig_filename"],
                "reason": f"image_read_failed: {e}",
            })
            continue

        try:
            merged_mask = load_merged_mask(mask_paths)
            info = compute_bbox_from_mask(merged_mask)
            xmin, ymin, xmax, ymax = info["bbox"]
        except Exception as e:
            skipped_rows.append({
                "split": split_name,
                "class_name": class_name,
                "orig_filename": record["orig_filename"],
                "reason": f"bbox_failed: {e}",
            })
            continue

        new_stem = build_new_stem(split_name, class_name, record["orig_stem"])
        new_filename = f"{new_stem}{record['orig_suffix']}"
        dst_image_path = JPEG_DIR / new_filename
        xml_path = ANN_DIR / f"{new_stem}.xml"

        if COPY_IMAGES:
            shutil.copy2(image_path, dst_image_path)

        write_voc_xml(
            xml_path=xml_path,
            image_filename=new_filename,
            image_abs_path=dst_image_path,
            width=width,
            height=height,
            depth=depth,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

        if MAKE_PREVIEWS and preview_counts[split_name] < MAX_PREVIEWS_PER_SPLIT:
            try:
                split_preview_dir = PREVIEW_DIR / split_name
                split_preview_dir.mkdir(parents=True, exist_ok=True)
                save_preview(
                    image_path=image_path,
                    preview_path=split_preview_dir / new_filename,
                    final_bbox=(xmin, ymin, xmax, ymax),
                    raw_bbox=info["raw_union_bbox"],
                    main_bbox=info["main_bbox"],
                )
                preview_counts[split_name] += 1
            except Exception:
                pass

        row = {
            "split": split_name,
            "class_name": class_name,
            "label": label,
            "orig_filename": record["orig_filename"],
            "orig_stem": record["orig_stem"],
            "new_stem": new_stem,
            "new_filename": new_filename,
            "width": width,
            "height": height,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "n_mask_files": int(record["n_mask_files"]),
            "mask_file_names": "|".join(Path(p).name for p in mask_paths),
        }
        row.update({
            "raw_union_bbox": str(info["raw_union_bbox"]),
            "main_bbox": str(info["main_bbox"]),
            "largest_area": info["largest_area"],
            "all_components": info["all_components"],
            "kept_components": info["kept_components"],
            "kept_component_ids": info["kept_component_ids"],
            "kept_reasons": info["kept_reasons"],
            "area_threshold": info["area_threshold"],
            "raw_area_ratio": info["raw_area_ratio"],
            "crop_mode": info["crop_mode"],
            "raw_square_side": info["raw_square_side"],
            "square_scale": info["square_scale"],
            "target_square_side": info["target_square_side"],
            "min_square_side": info["min_square_side"],
            "max_square_side": info["max_square_side"],
            "final_square_side": info["final_square_side"],
            "final_square_side_ratio": info["final_square_side_ratio"],
        })
        row.update(compute_qc_flags((xmin, ymin, xmax, ymax), width, height))
        ok_rows.append(row)

    return ok_rows, skipped_rows


def write_split_bbox_stats(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return

    def as_int(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    group_stats: Dict[str, Dict[str, List[float] | int]] = defaultdict(lambda: {
        "n_samples": 0,
        "box_area_ratio": [],
        "raw_area_ratio": [],
        "final_square_side_ratio": [],
        "touch_any": 0,
        "touch_both_lr": 0,
        "near_any_edge": 0,
        "too_small": 0,
        "too_large": 0,
        "near_full_image": 0,
    })

    for r in rows:
        split = str(r["split"])
        g = group_stats[split]
        g["n_samples"] = int(g["n_samples"]) + 1
        g["box_area_ratio"].append(float(r["box_area_ratio"]))
        g["raw_area_ratio"].append(float(r["raw_area_ratio"]))
        g["final_square_side_ratio"].append(float(r["final_square_side_ratio"]))
        g["touch_any"] = int(g["touch_any"]) + as_int(r["touch_any"])
        g["touch_both_lr"] = int(g["touch_both_lr"]) + as_int(r["touch_both_lr"])
        g["near_any_edge"] = int(g["near_any_edge"]) + as_int(r["near_any_edge"])
        g["too_small"] = int(g["too_small"]) + as_int(r["too_small"])
        g["too_large"] = int(g["too_large"]) + as_int(r["too_large"])
        g["near_full_image"] = int(g["near_full_image"]) + as_int(r["near_full_image"])

    out_rows: List[Dict] = []
    for split, g in sorted(group_stats.items()):
        n = int(g["n_samples"])
        box_area_ratio = np.array(g["box_area_ratio"], dtype=np.float64)
        raw_area_ratio = np.array(g["raw_area_ratio"], dtype=np.float64)
        side_ratio = np.array(g["final_square_side_ratio"], dtype=np.float64)

        out_rows.append({
            "split": split,
            "n_samples": n,
            "box_area_ratio_mean": round(float(box_area_ratio.mean()), 6),
            "box_area_ratio_median": round(float(np.median(box_area_ratio)), 6),
            "raw_area_ratio_mean": round(float(raw_area_ratio.mean()), 6),
            "raw_area_ratio_median": round(float(np.median(raw_area_ratio)), 6),
            "final_square_side_ratio_mean": round(float(side_ratio.mean()), 6),
            "final_square_side_ratio_median": round(float(np.median(side_ratio)), 6),
            "touch_any_sum": int(g["touch_any"]),
            "touch_any_rate": round(int(g["touch_any"]) / n, 6),
            "touch_both_lr_sum": int(g["touch_both_lr"]),
            "touch_both_lr_rate": round(int(g["touch_both_lr"]) / n, 6),
            "near_any_edge_sum": int(g["near_any_edge"]),
            "near_any_edge_rate": round(int(g["near_any_edge"]) / n, 6),
            "too_small_sum": int(g["too_small"]),
            "too_small_rate": round(int(g["too_small"]) / n, 6),
            "too_large_sum": int(g["too_large"]),
            "too_large_rate": round(int(g["too_large"]) / n, 6),
            "near_full_image_sum": int(g["near_full_image"]),
            "near_full_image_rate": round(int(g["near_full_image"]) / n, 6),
        })

    write_csv(path, out_rows)


def ensure_unique_new_stems(rows: List[Dict]) -> None:
    stems = [str(r["new_stem"]) for r in rows]
    if len(stems) != len(set(stems)):
        dup = [x for x in sorted(set(stems)) if stems.count(x) > 1][:20]
        raise RuntimeError(f"Duplicate new_stem detected, examples={dup}")


def main() -> None:
    print("=" * 100)
    print("BUSI lesion-only -> VOC ROI dataset builder (v3 square-consistent)")
    print("=" * 100)
    print(f"[INFO] SOURCE_ROOT = {SOURCE_ROOT}")
    print(f"[INFO] OUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"[INFO] INCLUDE_CLASSES = {INCLUDE_CLASSES}")
    print(f"[INFO] TEST_RATIO = {TEST_RATIO}, N_FOLDS = {N_FOLDS}, SPLIT_SEED = {SPLIT_SEED}")
    print(
        f"[INFO] v3 ROI: MIN_COMPONENT_AREA={MIN_COMPONENT_AREA}, MIN_RELATIVE_AREA={MIN_RELATIVE_AREA}, "
        f"NEAR_MARGIN_RATIO={NEAR_MARGIN_RATIO}, CENTROID_DIST_RATIO={CENTROID_DIST_RATIO}"
    )
    print(
        f"[INFO] square-consistent: MIN_FINAL_SIDE_RATIO={MIN_FINAL_SIDE_RATIO}, "
        f"MAX_FINAL_SIDE_RATIO={MAX_FINAL_SIDE_RATIO}, "
        f"scales=({SCALE_TINY}, {SCALE_SMALL}, {SCALE_MEDIUM}, {SCALE_LARGE})"
    )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SOURCE_ROOT 不存在: {SOURCE_ROOT}")

    reset_output_root()

    records, discovery_skipped = discover_busi_records()
    if not records:
        raise RuntimeError("未发现可用 BUSI lesion 样本。请检查目录结构。")

    print(f"[INFO] discovered usable lesion samples = {len(records)}")
    for cls in INCLUDE_CLASSES:
        cnt = sum(1 for r in records if r["class_name"] == cls)
        print(f"[INFO] class {cls:10s} -> {cnt}")

    trainval_indices, test_indices = build_stratified_trainval_test(records)
    folds = build_stratified_folds(trainval_indices, records, N_FOLDS)

    split_assignments: Dict[int, str] = {}
    for idx in trainval_indices:
        split_assignments[idx] = "trainval"
    for idx in test_indices:
        split_assignments[idx] = "test"

    split_rows: List[Dict] = []
    trainval_set = set(trainval_indices)
    test_set = set(test_indices)
    fold_train_sets = [set(fd["train"]) for fd in folds]
    fold_val_sets = [set(fd["val"]) for fd in folds]

    for idx, r in enumerate(records):
        row = {
            "record_idx": idx,
            "class_name": r["class_name"],
            "label": r["label"],
            "orig_filename": r["orig_filename"],
            "image_path": r["image_path"],
            "n_mask_files": r["n_mask_files"],
            "mask_paths": "|".join(r["mask_paths"]),
            "split": "trainval" if idx in trainval_set else ("test" if idx in test_set else "unknown"),
        }
        for fold_id in range(N_FOLDS):
            row[f"fold{fold_id}_train"] = int(idx in fold_train_sets[fold_id])
            row[f"fold{fold_id}_val"] = int(idx in fold_val_sets[fold_id])
        split_rows.append(row)

    ok_rows, process_skipped = process_records(records, split_assignments)
    all_skipped = discovery_skipped + process_skipped

    ensure_unique_new_stems(ok_rows)

    trainval_stems = [r["new_stem"] for r in ok_rows if r["split"] == "trainval"]
    test_stems = [r["new_stem"] for r in ok_rows if r["split"] == "test"]
    write_lines(IMAGESETS_MAIN_DIR / "trainval.txt", trainval_stems)
    write_lines(IMAGESETS_MAIN_DIR / "test.txt", test_stems)

    idx_to_new_stem: Dict[int, str] = {}
    lookup = {(r["orig_filename"], r["split"]): r["new_stem"] for r in ok_rows}
    for idx, r in enumerate(records):
        sp = split_assignments[idx]
        key = (r["orig_filename"], sp)
        if key in lookup:
            idx_to_new_stem[idx] = lookup[key]

    for fold_id, fold in enumerate(folds):
        fold_train = [idx_to_new_stem[i] for i in fold["train"] if i in idx_to_new_stem]
        fold_val = [idx_to_new_stem[i] for i in fold["val"] if i in idx_to_new_stem]
        write_lines(IMAGESETS_MAIN_DIR / f"fold{fold_id}_train.txt", fold_train)
        write_lines(IMAGESETS_MAIN_DIR / f"fold{fold_id}_val.txt", fold_val)
        print(f"[INFO] fold{fold_id}: train={len(fold_train)} | val={len(fold_val)}")

    write_csv(MANIFEST_DIR / "original_records.csv", split_rows)
    write_csv(MANIFEST_DIR / "bbox_manifest.csv", ok_rows)
    write_csv(MANIFEST_DIR / "missing_or_skipped.csv", all_skipped)

    label_rows = []
    for r in ok_rows:
        label_rows.append({
            "new_filename": r["new_filename"],
            "label": r["label"],
            "split": r["split"],
            "class_name": r["class_name"],
            "orig_filename": r["orig_filename"],
            "new_stem": r["new_stem"],
        })
    write_csv(MANIFEST_DIR / "label_manifest.csv", label_rows)
    write_split_bbox_stats(MANIFEST_DIR / "split_bbox_stats.csv", ok_rows)

    split_json = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "include_classes": INCLUDE_CLASSES,
        "class_to_label": CLASS_TO_LABEL,
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "split_seed": SPLIT_SEED,
        "n_records_total": len(records),
        "n_ok": len(ok_rows),
        "n_skipped": len(all_skipped),
        "trainval_indices": trainval_indices,
        "test_indices": test_indices,
        "folds": folds,
    }
    with open(MANIFEST_DIR / "split_assignments.json", "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)

    print("-" * 100)
    print(f"[DONE] usable converted samples = {len(ok_rows)}")
    print(f"[DONE] skipped samples          = {len(all_skipped)}")
    print(f"[DONE] JPEGImages              = {JPEG_DIR}")
    print(f"[DONE] Annotations             = {ANN_DIR}")
    print(f"[DONE] ImageSets/Main          = {IMAGESETS_MAIN_DIR}")
    print(f"[DONE] Manifests               = {MANIFEST_DIR}")
    print(f"[DONE] BBox previews           = {PREVIEW_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
