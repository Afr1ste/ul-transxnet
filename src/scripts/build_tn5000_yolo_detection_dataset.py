import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def read_split(path: Path) -> list[str]:
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_voc_boxes(xml_path: Path) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
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


def build_dataset(src_root: Path, out_root: Path, overwrite: bool) -> dict:
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_root = src_root / "ImageSets" / "Main"
    image_root = src_root / "JPEGImages"
    ann_root = src_root / "Annotations"

    summary: dict[str, object] = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "class_names": {0: "lesion"},
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        ids = read_split(split_root / f"{split}.txt")
        split_summary = {
            "images": 0,
            "objects": 0,
            "missing_images": [],
            "missing_annotations": [],
            "empty_annotations": [],
            "link_modes": {},
        }
        for image_id in ids:
            src_img = image_root / f"{image_id}.jpg"
            xml_path = ann_root / f"{image_id}.xml"
            if not src_img.exists():
                split_summary["missing_images"].append(image_id)
                continue
            if not xml_path.exists():
                split_summary["missing_annotations"].append(image_id)
                continue

            _, _, boxes = parse_voc_boxes(xml_path)
            if not boxes:
                split_summary["empty_annotations"].append(image_id)

            dst_img = out_root / "images" / split / src_img.name
            mode = link_or_copy(src_img, dst_img)
            split_summary["link_modes"][mode] = split_summary["link_modes"].get(mode, 0) + 1

            label_lines = [f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}" for xc, yc, bw, bh in boxes]
            label_path = out_root / "labels" / split / f"{image_id}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            split_summary["images"] += 1
            split_summary["objects"] += len(boxes)

        summary["splits"][split] = split_summary

    yaml_text = "\n".join(
        [
            f"path: {out_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: lesion",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO lesion-detection dataset from TN5000 VOC annotations.")
    parser.add_argument("--src-root", type=Path, default=Path("TN5000_forReview"))
    parser.add_argument("--out-root", type=Path, default=Path("detector_datasets/tn5000_yolo_lesion_v1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    summary = build_dataset(src_root, out_root, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
