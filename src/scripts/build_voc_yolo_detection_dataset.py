from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_split(path: Path) -> list[str]:
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def detect_splits(split_root: Path) -> list[str]:
    split_names = [p.stem for p in sorted(split_root.glob("*.txt"))]
    # Keep train/val/test-like splits and fold splits; skip aggregate helper files by default.
    keep: list[str] = []
    for name in split_names:
        if name in {"train", "val", "test"} or re.match(r"fold\d+_(train|val)$", name):
            keep.append(name)
    return keep


def image_size_from_file(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(f"Missing VOC <size> and Pillow is unavailable: {image_path}") from exc
    with Image.open(image_path) as img:
        return img.size


def find_image(image_root: Path, image_id: str, xml_root: ET.Element | None = None) -> Path | None:
    candidates: list[Path] = []
    filename = xml_root.findtext("filename") if xml_root is not None else None
    if filename:
        candidates.append(image_root / filename)
    raw = Path(image_id)
    if raw.suffix:
        candidates.append(image_root / raw.name)
    else:
        candidates.extend(image_root / f"{image_id}{ext}" for ext in IMAGE_EXTS)
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_voc_boxes(xml_path: Path, image_path: Path | None = None) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None:
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
    elif image_path is not None:
        width, height = image_size_from_file(image_path)
    else:
        raise ValueError(f"Missing size in {xml_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")

    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width - 1.0))
        xmax = max(0.0, min(xmax, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        ymax = max(0.0, min(ymax, height - 1.0))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        boxes.append((x_center, y_center, box_w, box_h))
    return width, height, boxes


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def yaml_text(out_root: Path, train_split: str, val_split: str, test_split: str | None, class_name: str) -> str:
    lines = [
        f"path: {out_root.as_posix()}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
    ]
    if test_split:
        lines.append(f"test: images/{test_split}")
    lines.extend(["names:", f"  0: {class_name}", ""])
    return "\n".join(lines)


def write_yaml_files(out_root: Path, splits: list[str], class_name: str) -> list[str]:
    yaml_files: list[str] = []
    split_set = set(splits)
    if {"train", "val"}.issubset(split_set):
        test_split = "test" if "test" in split_set else None
        path = out_root / "data.yaml"
        path.write_text(yaml_text(out_root, "train", "val", test_split, class_name), encoding="utf-8")
        yaml_files.append(str(path))

    fold_ids = sorted(
        {
            m.group(1)
            for split in splits
            for m in [re.match(r"fold(\d+)_(train|val)$", split)]
            if m is not None
        },
        key=int,
    )
    for fold_id in fold_ids:
        train_split = f"fold{fold_id}_train"
        val_split = f"fold{fold_id}_val"
        if train_split not in split_set or val_split not in split_set:
            continue
        test_split = "test" if "test" in split_set else None
        path = out_root / f"data_fold{fold_id}.yaml"
        path.write_text(yaml_text(out_root, train_split, val_split, test_split, class_name), encoding="utf-8")
        yaml_files.append(str(path))
    return yaml_files


def build_dataset(src_root: Path, out_root: Path, splits: list[str] | None, overwrite: bool, class_name: str) -> dict:
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_root = src_root / "ImageSets" / "Main"
    image_root = src_root / "JPEGImages"
    ann_root = src_root / "Annotations"
    if splits is None:
        splits = detect_splits(split_root)
    if not splits:
        raise RuntimeError(f"No usable split files found under {split_root}")

    summary: dict[str, object] = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "class_names": {0: class_name},
        "splits_requested": splits,
        "splits": {},
        "yaml_files": [],
    }

    for split in splits:
        split_file = split_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        ids = read_split(split_file)
        split_summary: dict[str, object] = {
            "ids": len(ids),
            "images": 0,
            "objects": 0,
            "missing_images": [],
            "missing_annotations": [],
            "empty_annotations": [],
            "link_modes": {},
        }
        for image_id in ids:
            xml_path = ann_root / f"{Path(image_id).stem}.xml"
            if not xml_path.exists():
                split_summary["missing_annotations"].append(image_id)
                continue

            xml_root = ET.parse(xml_path).getroot()
            src_img = find_image(image_root, image_id, xml_root)
            if src_img is None:
                split_summary["missing_images"].append(image_id)
                continue

            _, _, boxes = parse_voc_boxes(xml_path, src_img)
            if not boxes:
                split_summary["empty_annotations"].append(image_id)

            dst_img = out_root / "images" / split / src_img.name
            mode = link_or_copy(src_img, dst_img)
            link_modes = split_summary["link_modes"]
            link_modes[mode] = link_modes.get(mode, 0) + 1

            label_lines = [f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}" for xc, yc, bw, bh in boxes]
            label_path = out_root / "labels" / split / f"{src_img.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            split_summary["images"] += 1
            split_summary["objects"] += len(boxes)

        summary["splits"][split] = split_summary

    summary["yaml_files"] = write_yaml_files(out_root, splits, class_name)
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-class YOLO lesion-detection dataset from VOC annotations.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split names without .txt; default auto-detects folds/test.")
    parser.add_argument("--class-name", default="lesion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_dataset(
        src_root=args.src_root.resolve(),
        out_root=args.out_root.resolve(),
        splits=args.splits,
        overwrite=args.overwrite,
        class_name=args.class_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
