#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUL (Annotated Ultrasound Liver) -> VOC-style ROI binary dataset builder

目标
----
将 AUL 的 Benign / Malignant 两个压缩包（或已解压目录）转换成你当前训练脚本可直接读取的
VOC 风格 ROI 分类数据集：

- JPEGImages/
- Annotations/           (PASCAL VOC XML, 单框或多框 union 后单框)
- ImageSets/Main/
- manifests/label_manifest.csv
- manifests/split_stats.csv
- bbox_previews/

标签定义
--------
- benign    -> 0
- malignant -> 1

数据来源结构（原始）
------------------
Benign/
  image/*.jpg
  segmentation/mass/*.json
  segmentation/outline/*.json
  segmentation/liver/*.json
Malignant/
  image/*.jpg
  segmentation/mass/*.json
  segmentation/outline/*.json
  segmentation/liver/*.json

当前脚本默认只依赖 mass 多边形生成 ROI。
outline / liver 用于 QA 统计与预览，不作为硬依赖。

切分协议
--------
1) 先做 image-level stratified trainval / test split
2) 再在 trainval 上做 5-fold stratified split

注意
----
AUL 公开包里未见 patient_id 元信息，因此这里无法做 patient-level split，
只能退而求其次使用 image-level split。论文中如需说明，必须明确写出这一限制。
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedKFold, train_test_split

# =========================
# 用户配置
# =========================
PROJECT_ROOT = Path(r"<LOCAL_THYROID_ROOT>")
BENIGN_SOURCE = Path(os.environ.get("AUL_BENIGN_SOURCE", str(PROJECT_ROOT / "Benign.zip")))
MALIGNANT_SOURCE = Path(os.environ.get("AUL_MALIGNANT_SOURCE", str(PROJECT_ROOT / "Malignant.zip")))
OUTPUT_ROOT = Path(os.environ.get("AUL_OUTPUT_ROOT", str(PROJECT_ROOT / "aul" / "aul_voc_roi_v1")))

# 若你已经手动解压，也可以把上面两个 SOURCE 指向目录：
#   PROJECT_ROOT / "Benign"
#   PROJECT_ROOT / "Malignant"

TEST_RATIO = 0.20
N_FOLDS = 5
SPLIT_SEED = 42
USE_HARDLINKS = True
COPY_BBOX_PREVIEWS = True

# ROI / bbox 规则
BBOX_EXPAND_RATIO = 0.30   # 仅用于 preview 显示；训练脚本里仍可再 expand
MIN_BOX_SIDE = 16
FORCE_SQUARE_BOX = True
SQUARE_CONTEXT_RATIO = 1.15    # 在最小包围框基础上再放大一点，得到正方形 ROI

# 预览数量
MAX_PREVIEW_PER_CLASS = 25

LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
}

CLASS_NAME_MAP = {
    0: "benign",
    1: "malignant",
}


# =========================
# 数据结构
# =========================
@dataclass
class Sample:
    cls_name: str                    # benign / malignant
    label: int                       # 0 / 1
    image_id: str                    # e.g. benign_000123
    source_image_rel: str            # image/123.jpg
    mass_rel: str                    # segmentation/mass/123.json
    outline_rel: Optional[str]       # segmentation/outline/123.json
    liver_rel: Optional[str]         # segmentation/liver/123.json
    bbox_xyxy: Tuple[int, int, int, int]
    width: int
    height: int


# =========================
# 通用工具
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_unlink(p: Path) -> None:
    try:
        if p.exists() or p.is_symlink():
            p.unlink()
    except Exception:
        pass


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def polygon_bbox(points: Sequence[Sequence[float]], width: int, height: int) -> Tuple[int, int, int, int]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0, int(math.floor(min(xs))))
    y1 = max(0, int(math.floor(min(ys))))
    x2 = min(width - 1, int(math.ceil(max(xs))))
    y2 = min(height - 1, int(math.ceil(max(ys))))
    return x1, y1, x2, y2


def clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def maybe_square_box(box: Tuple[int, int, int, int], width: int, height: int,
                     min_side: int = MIN_BOX_SIDE,
                     square_context_ratio: float = SQUARE_CONTEXT_RATIO) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(x2 - x1 + 1, min_side)
    bh = max(y2 - y1 + 1, min_side)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if FORCE_SQUARE_BOX:
        side = max(bw, bh)
        side = max(min_side, int(round(side * square_context_ratio)))
        half = side / 2.0
        nx1 = int(round(cx - half))
        ny1 = int(round(cy - half))
        nx2 = int(round(cx + half)) - 1
        ny2 = int(round(cy + half)) - 1
    else:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2

    return clip_box((nx1, ny1, nx2, ny2), width, height)


def expand_box(box: Tuple[int, int, int, int], width: int, height: int,
               ratio: float = BBOX_EXPAND_RATIO) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = bw * (1 + 2 * ratio)
    nh = bh * (1 + 2 * ratio)
    nx1 = int(round(cx - nw / 2.0))
    ny1 = int(round(cy - nh / 2.0))
    nx2 = int(round(cx + nw / 2.0)) - 1
    ny2 = int(round(cy + nh / 2.0)) - 1
    return clip_box((nx1, ny1, nx2, ny2), width, height)


# =========================
# 读取来源（zip 或目录）
# =========================
class SourceReader:
    def __init__(self, root_or_zip: Path):
        self.path = Path(root_or_zip)
        self.is_zip = self.path.suffix.lower() == ".zip"
        self.zf: Optional[zipfile.ZipFile] = None
        if self.is_zip:
            self.zf = zipfile.ZipFile(self.path, "r")
            # 取顶层目录名，如 Benign/ 或 Malignant/
            top_dirs = [n.split("/")[0] for n in self.zf.namelist() if "/" in n and not n.startswith("__MACOSX/")]
            self.root_prefix = sorted(set(top_dirs))[0] + "/"
        else:
            self.root_prefix = ""

    def close(self):
        if self.zf is not None:
            self.zf.close()

    def _norm_rel(self, rel: str) -> str:
        rel = rel.replace("\\", "/").lstrip("/")
        if rel.startswith(self.root_prefix):
            return rel
        return self.root_prefix + rel

    def exists(self, rel: str) -> bool:
        rel = self._norm_rel(rel)
        if self.is_zip:
            assert self.zf is not None
            try:
                self.zf.getinfo(rel)
                return True
            except KeyError:
                return False
        return (self.path / rel).exists()

    def read_bytes(self, rel: str) -> bytes:
        rel = self._norm_rel(rel)
        if self.is_zip:
            assert self.zf is not None
            return self.zf.read(rel)
        return (self.path / rel).read_bytes()

    def read_json(self, rel: str):
        return json.loads(self.read_bytes(rel).decode("utf-8"))

    def iter_image_ids(self) -> List[str]:
        results = []
        if self.is_zip:
            assert self.zf is not None
            for n in self.zf.namelist():
                if n.startswith("__MACOSX/"):
                    continue
                if n.startswith(self.root_prefix + "image/") and n.lower().endswith(".jpg"):
                    results.append(Path(n).stem)
        else:
            img_dir = self.path / self.root_prefix / "image"
            for p in sorted(img_dir.glob("*.jpg")):
                results.append(p.stem)
        results = sorted(set(results), key=lambda x: int(x))
        return results

    def export_file(self, rel: str, dst: Path, use_hardlink: bool = False) -> None:
        ensure_dir(dst.parent)
        rel = self._norm_rel(rel)
        if self.is_zip:
            # zip 内文件无法硬链接，直接写出
            data = self.read_bytes(rel)
            dst.write_bytes(data)
        else:
            src = self.path / rel
            if use_hardlink:
                safe_unlink(dst)
                try:
                    os.link(src, dst)
                    return
                except Exception:
                    pass
            shutil.copy2(src, dst)


# =========================
# 核心解析
# =========================
def load_samples(reader: SourceReader, cls_name: str) -> Tuple[List[Sample], List[Dict]]:
    label = LABEL_MAP[cls_name]
    samples: List[Sample] = []
    issues: List[Dict] = []

    for rid in reader.iter_image_ids():
        img_rel = f"image/{rid}.jpg"
        mass_rel = f"segmentation/mass/{rid}.json"
        outline_rel = f"segmentation/outline/{rid}.json"
        liver_rel = f"segmentation/liver/{rid}.json"

        if not reader.exists(mass_rel):
            issues.append({
                "class_name": cls_name,
                "raw_id": rid,
                "issue": "missing_mass_json",
            })
            continue

        try:
            img_bytes = reader.read_bytes(img_rel)
            with Image.open(io_from_bytes(img_bytes)) as im:
                width, height = im.size
        except Exception as e:
            issues.append({
                "class_name": cls_name,
                "raw_id": rid,
                "issue": f"bad_image:{repr(e)}",
            })
            continue

        try:
            mass_pts = reader.read_json(mass_rel)
            if not isinstance(mass_pts, list) or len(mass_pts) < 3:
                raise ValueError("mass polygon invalid")
        except Exception as e:
            issues.append({
                "class_name": cls_name,
                "raw_id": rid,
                "issue": f"bad_mass_json:{repr(e)}",
            })
            continue

        bbox = polygon_bbox(mass_pts, width, height)
        bbox = maybe_square_box(bbox, width, height)

        image_id = f"{cls_name}_{int(rid):06d}"
        samples.append(Sample(
            cls_name=cls_name,
            label=label,
            image_id=image_id,
            source_image_rel=img_rel,
            mass_rel=mass_rel,
            outline_rel=outline_rel if reader.exists(outline_rel) else None,
            liver_rel=liver_rel if reader.exists(liver_rel) else None,
            bbox_xyxy=bbox,
            width=width,
            height=height,
        ))

    return samples, issues


def io_from_bytes(b: bytes):
    import io
    return io.BytesIO(b)


# =========================
# VOC 输出
# =========================
def write_voc_xml(path: Path, filename: str, width: int, height: int,
                  bbox: Tuple[int, int, int, int], class_name: str) -> None:
    x1, y1, x2, y2 = bbox
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "JPEGImages"
    ET.SubElement(ann, "filename").text = filename

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    obj = ET.SubElement(ann, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bnd = ET.SubElement(obj, "bndbox")
    ET.SubElement(bnd, "xmin").text = str(x1)
    ET.SubElement(bnd, "ymin").text = str(y1)
    ET.SubElement(bnd, "xmax").text = str(x2)
    ET.SubElement(bnd, "ymax").text = str(y2)

    tree = ET.ElementTree(ann)
    ensure_dir(path.parent)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def save_preview(reader: SourceReader, sample: Sample, out_path: Path) -> None:
    img = Image.open(io_from_bytes(reader.read_bytes(sample.source_image_rel))).convert("RGB")
    draw = ImageDraw.Draw(img)

    # mass polygon
    try:
        mass_pts = reader.read_json(sample.mass_rel)
        draw.polygon([(float(x), float(y)) for x, y in mass_pts], outline=(255, 0, 0), width=3)
    except Exception:
        pass

    # liver polygon
    if sample.liver_rel is not None:
        try:
            liver_pts = reader.read_json(sample.liver_rel)
            draw.polygon([(float(x), float(y)) for x, y in liver_pts], outline=(0, 255, 0), width=2)
        except Exception:
            pass

    # bbox
    bx = expand_box(sample.bbox_xyxy, sample.width, sample.height, ratio=0.0)
    draw.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=(0, 170, 255), width=3)

    ensure_dir(out_path.parent)
    img.save(out_path)


# =========================
# 主流程
# =========================
def main() -> None:
    print("=" * 100)
    print("AUL -> VOC ROI dataset builder (v1)")
    print("=" * 100)
    print(f"[INFO] BENIGN_SOURCE    = {BENIGN_SOURCE}")
    print(f"[INFO] MALIGNANT_SOURCE = {MALIGNANT_SOURCE}")
    print(f"[INFO] OUTPUT_ROOT      = {OUTPUT_ROOT}")
    print(f"[INFO] TEST_RATIO={TEST_RATIO}, N_FOLDS={N_FOLDS}, SPLIT_SEED={SPLIT_SEED}")
    print(f"[INFO] USE_HARDLINKS={USE_HARDLINKS}, FORCE_SQUARE_BOX={FORCE_SQUARE_BOX}, SQUARE_CONTEXT_RATIO={SQUARE_CONTEXT_RATIO}")

    jpeg_dir = OUTPUT_ROOT / "JPEGImages"
    ann_dir = OUTPUT_ROOT / "Annotations"
    split_dir = OUTPUT_ROOT / "ImageSets" / "Main"
    manifest_dir = OUTPUT_ROOT / "manifests"
    preview_dir = OUTPUT_ROOT / "bbox_previews"
    for d in [jpeg_dir, ann_dir, split_dir, manifest_dir, preview_dir]:
        ensure_dir(d)

    benign_reader = SourceReader(BENIGN_SOURCE)
    malignant_reader = SourceReader(MALIGNANT_SOURCE)
    try:
        benign_samples, benign_issues = load_samples(benign_reader, "benign")
        malignant_samples, malignant_issues = load_samples(malignant_reader, "malignant")
    finally:
        # 预览阶段还要用 reader，所以不能提前 close。后面再关。
        pass

    all_samples = benign_samples + malignant_samples
    issues = benign_issues + malignant_issues

    print(f"[INFO] benign usable    = {len(benign_samples)}")
    print(f"[INFO] malignant usable = {len(malignant_samples)}")
    print(f"[INFO] total usable     = {len(all_samples)}")
    print(f"[INFO] total issues     = {len(issues)}")

    # 输出图像 + XML + manifest
    manifest_rows: List[Dict] = []
    preview_counter = {"benign": 0, "malignant": 0}

    for s in all_samples:
        reader = benign_reader if s.cls_name == "benign" else malignant_reader
        out_img_name = f"{s.image_id}.jpg"
        out_xml_name = f"{s.image_id}.xml"

        reader.export_file(s.source_image_rel, jpeg_dir / out_img_name, use_hardlink=USE_HARDLINKS)
        write_voc_xml(ann_dir / out_xml_name, out_img_name, s.width, s.height, s.bbox_xyxy, s.cls_name)

        manifest_rows.append({
            "image_id": s.image_id,
            "filename": out_img_name,
            "xml_filename": out_xml_name,
            "label": s.label,
            "class_name": s.cls_name,
            "width": s.width,
            "height": s.height,
            "bbox_xmin": s.bbox_xyxy[0],
            "bbox_ymin": s.bbox_xyxy[1],
            "bbox_xmax": s.bbox_xyxy[2],
            "bbox_ymax": s.bbox_xyxy[3],
            "source_rel": s.source_image_rel,
            "mass_rel": s.mass_rel,
            "outline_rel": s.outline_rel or "",
            "liver_rel": s.liver_rel or "",
        })

        if COPY_BBOX_PREVIEWS and preview_counter[s.cls_name] < MAX_PREVIEW_PER_CLASS:
            save_preview(reader, s, preview_dir / s.cls_name / f"{s.image_id}.jpg")
            preview_counter[s.cls_name] += 1

    # split
    image_ids = [s.image_id for s in all_samples]
    labels = [s.label for s in all_samples]
    trainval_ids, test_ids, trainval_y, test_y = train_test_split(
        image_ids,
        labels,
        test_size=TEST_RATIO,
        random_state=SPLIT_SEED,
        shuffle=True,
        stratify=labels,
    )

    def write_ids(path: Path, ids: List[str]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            for x in ids:
                f.write(f"{x}\n")

    write_ids(split_dir / "trainval.txt", sorted(trainval_ids))
    write_ids(split_dir / "test.txt", sorted(test_ids))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    trainval_ids_arr = list(trainval_ids)
    trainval_y_arr = list(trainval_y)

    fold_stats_rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(trainval_ids_arr, trainval_y_arr)):
        fold_train = [trainval_ids_arr[i] for i in tr_idx]
        fold_val = [trainval_ids_arr[i] for i in va_idx]
        write_ids(split_dir / f"fold{fold_idx}_train.txt", sorted(fold_train))
        write_ids(split_dir / f"fold{fold_idx}_val.txt", sorted(fold_val))

        tr_labels = [trainval_y_arr[i] for i in tr_idx]
        va_labels = [trainval_y_arr[i] for i in va_idx]
        tr_b = sum(int(x == 0) for x in tr_labels)
        tr_m = sum(int(x == 1) for x in tr_labels)
        va_b = sum(int(x == 0) for x in va_labels)
        va_m = sum(int(x == 1) for x in va_labels)
        fold_stats_rows.append({
            "split": f"fold{fold_idx}",
            "train_total": len(fold_train),
            "train_benign": tr_b,
            "train_malignant": tr_m,
            "val_total": len(fold_val),
            "val_benign": va_b,
            "val_malignant": va_m,
        })
        print(f"[INFO] fold{fold_idx}: train={len(fold_train)} {{0:{tr_b},1:{tr_m}}} | val={len(fold_val)} {{0:{va_b},1:{va_m}}}")

    trainval_b = sum(int(y == 0) for y in trainval_y)
    trainval_m = sum(int(y == 1) for y in trainval_y)
    test_b = sum(int(y == 0) for y in test_y)
    test_m = sum(int(y == 1) for y in test_y)

    write_csv(manifest_dir / "label_manifest.csv", manifest_rows, fieldnames=[
        "image_id", "filename", "xml_filename", "label", "class_name",
        "width", "height", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "source_rel", "mass_rel", "outline_rel", "liver_rel"
    ])

    split_stats = [
        {"split": "all", "total": len(all_samples), "benign": len(benign_samples), "malignant": len(malignant_samples)},
        {"split": "trainval", "total": len(trainval_ids), "benign": trainval_b, "malignant": trainval_m},
        {"split": "test", "total": len(test_ids), "benign": test_b, "malignant": test_m},
    ]
    write_csv(manifest_dir / "split_stats.csv", split_stats, fieldnames=["split", "total", "benign", "malignant"])
    write_csv(manifest_dir / "fold_stats.csv", fold_stats_rows, fieldnames=["split", "train_total", "train_benign", "train_malignant", "val_total", "val_benign", "val_malignant"])
    if issues:
        write_csv(manifest_dir / "issues.csv", issues, fieldnames=["class_name", "raw_id", "issue"])
    else:
        write_csv(manifest_dir / "issues.csv", [], fieldnames=["class_name", "raw_id", "issue"])

    meta = {
        "dataset": "AUL (Annotated Ultrasound Liver)",
        "task": "binary_classification",
        "label_map": {"benign": 0, "malignant": 1},
        "source_benign": str(BENIGN_SOURCE),
        "source_malignant": str(MALIGNANT_SOURCE),
        "output_root": str(OUTPUT_ROOT),
        "test_ratio": TEST_RATIO,
        "n_folds": N_FOLDS,
        "split_seed": SPLIT_SEED,
        "use_hardlinks": USE_HARDLINKS,
        "force_square_box": FORCE_SQUARE_BOX,
        "square_context_ratio": SQUARE_CONTEXT_RATIO,
        "notes": [
            "ROI box is generated from mass polygon only.",
            "Image-level stratified split is used because no patient_id metadata is available in the public package.",
            "Three malignant images miss liver polygon, but this does not affect ROI generation because mass polygon exists.",
        ],
    }
    save_json(manifest_dir / "dataset_meta.json", meta)

    benign_reader.close()
    malignant_reader.close()

    print("-" * 100)
    print(f"[DONE] total usable samples = {len(all_samples)}")
    print(f"[DONE] benign / malignant  = {len(benign_samples)} / {len(malignant_samples)}")
    print(f"[DONE] trainval / test     = {len(trainval_ids)} / {len(test_ids)}")
    print(f"[DONE] JPEGImages         = {jpeg_dir}")
    print(f"[DONE] Annotations        = {ann_dir}")
    print(f"[DONE] ImageSets/Main     = {split_dir}")
    print(f"[DONE] manifests          = {manifest_dir}")
    print(f"[DONE] bbox_previews      = {preview_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
