"""Audit BUSI duplicate and near-duplicate images without changing labels.

The output is intended as a provenance/limitation artifact for the manuscript.
It does not rewrite split files or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "busi" / "busi_voc_v3_square_consistent"
DEFAULT_OUT = PROJECT_ROOT / "eval_reports" / "busi_duplicate_audit_20260510"


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_label_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {Path(row["new_filename"]).stem: row for row in csv.DictReader(f)}


def parse_xml(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    filename = root.findtext("filename", default=path.with_suffix(".png").name)
    object_name = root.findtext("object/name", default="")
    bbox_node = root.find("object/bndbox")
    bbox = None
    if bbox_node is not None:
        bbox = tuple(int(float(bbox_node.findtext(k, "0"))) for k in ("xmin", "ymin", "xmax", "ymax"))
    return {"filename": filename, "object_name": object_name, "bbox": bbox}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as img:
        gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def ahash(path: Path, size: int = 8) -> int:
    with Image.open(path) as img:
        gray = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    mean_val = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | int(px >= mean_val)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def split_name(sample_id: str, trainval: set[str], test: set[str]) -> str:
    if sample_id in test:
        return "test"
    if sample_id in trainval:
        return "trainval"
    return "unknown"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--near-threshold", type=int, default=5)
    args = p.parse_args()

    root = args.root
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    split_dir = root / "ImageSets" / "Main"
    label_manifest = load_label_manifest(root / "manifests" / "label_manifest.csv")
    trainval = read_ids(split_dir / "trainval.txt")
    test = read_ids(split_dir / "test.txt")

    rows: list[dict[str, object]] = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        sample_id = xml_path.stem
        rec = parse_xml(xml_path)
        manifest_row = label_manifest.get(sample_id, {})
        if manifest_row:
            diagnostic_label = int(manifest_row["label"])
            class_name = manifest_row.get("class_name", "")
            original_filename = manifest_row.get("orig_filename", "")
        else:
            # Fallback for non-BUSI VOC trees where object/name stores the class.
            diagnostic_label = int(rec["object_name"])
            class_name = "benign" if diagnostic_label == 0 else "malignant"
            original_filename = ""
        image_path = img_dir / str(rec["filename"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
        rows.append(
            {
                "sample_id": sample_id,
                "split": split_name(sample_id, trainval, test),
                "label": diagnostic_label,
                "class_name": class_name,
                "xml_object_name": rec["object_name"],
                "orig_filename": original_filename,
                "filename": rec["filename"],
                "width": width,
                "height": height,
                "bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
                "dhash64": f"{dhash(image_path):016x}",
                "ahash64": f"{ahash(image_path):016x}",
                "image_path": str(image_path),
            }
        )

    exact_groups = defaultdict(list)
    for row in rows:
        exact_groups[row["sha256"]].append(row)
    exact_pairs: list[dict[str, object]] = []
    for group in exact_groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                exact_pairs.append(pair_row(a, b, 0, 0, "exact_sha256"))

    near_pairs: list[dict[str, object]] = []
    # O(n^2) is fine for BUSI-scale data.
    parsed = [
        (
            row,
            int(str(row["dhash64"]), 16),
            int(str(row["ahash64"]), 16),
        )
        for row in rows
    ]
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            a, ad, aa = parsed[i]
            b, bd, ba = parsed[j]
            dd = hamming(ad, bd)
            ah = hamming(aa, ba)
            if dd <= args.near_threshold and ah <= args.near_threshold:
                near_pairs.append(pair_row(a, b, dd, ah, "near_hash"))

    cross_split = [r for r in near_pairs + exact_pairs if r["split_a"] != r["split_b"]]
    label_conflict = [r for r in near_pairs + exact_pairs if r["label_a"] != r["label_b"]]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "busi_image_hash_manifest.csv", rows)
    write_csv(out_dir / "busi_exact_duplicate_pairs.csv", exact_pairs)
    write_csv(out_dir / "busi_near_duplicate_pairs.csv", near_pairs)
    write_csv(out_dir / "busi_cross_split_near_duplicate_pairs.csv", cross_split)
    write_csv(out_dir / "busi_label_conflict_duplicate_pairs.csv", label_conflict)

    summary = {
        "root": str(root),
        "n_images": len(rows),
        "trainval": len(trainval),
        "test": len(test),
        "near_threshold": args.near_threshold,
        "label_source": str(root / "manifests" / "label_manifest.csv") if label_manifest else "VOC XML object/name",
        "exact_duplicate_pairs": len(exact_pairs),
        "near_duplicate_pairs": len(near_pairs),
        "cross_split_duplicate_or_near_pairs": len(cross_split),
        "label_conflict_duplicate_or_near_pairs": len(label_conflict),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# BUSI Duplicate and Label-Noise Audit",
        "",
        "This audit is descriptive only. It does not modify labels, splits, or reported main metrics.",
        "",
        f"- BUSI VOC root: `{root}`",
        f"- Images audited: {len(rows)}",
        f"- Trainval IDs: {len(trainval)}",
        f"- Fixed test IDs: {len(test)}",
        f"- Exact duplicate pairs by SHA-256: {len(exact_pairs)}",
        f"- Near-duplicate pairs by dHash/aHash threshold <= {args.near_threshold}: {len(near_pairs)}",
        f"- Cross-split duplicate/near-duplicate pairs: {len(cross_split)}",
        f"- Label-conflict duplicate/near-duplicate pairs: {len(label_conflict)}",
        f"- Diagnostic label source: `{summary['label_source']}`",
        "- Note: VOC XML `object/name` is not used as the BUSI diagnostic label in this audit when `label_manifest.csv` is present.",
        "",
        "Outputs:",
        "- `busi_image_hash_manifest.csv`",
        "- `busi_exact_duplicate_pairs.csv`",
        "- `busi_near_duplicate_pairs.csv`",
        "- `busi_cross_split_near_duplicate_pairs.csv`",
        "- `busi_label_conflict_duplicate_pairs.csv`",
    ]
    if cross_split:
        md += ["", "First cross-split candidates:"]
        for row in cross_split[:10]:
            md.append(
                f"- {row['sample_id_a']} ({row['split_a']}, y={row['label_a']}) vs "
                f"{row['sample_id_b']} ({row['split_b']}, y={row['label_b']}), "
                f"dHash={row['dhash_hamming']}, aHash={row['ahash_hamming']}"
            )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def pair_row(a: dict[str, object], b: dict[str, object], dh: int, ah: int, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "sample_id_a": a["sample_id"],
        "split_a": a["split"],
        "label_a": a["label"],
        "sample_id_b": b["sample_id"],
        "split_b": b["split"],
        "label_b": b["label"],
        "dhash_hamming": dh,
        "ahash_hamming": ah,
        "same_label": a["label"] == b["label"],
        "same_split": a["split"] == b["split"],
        "sha256_a": a["sha256"],
        "sha256_b": b["sha256"],
        "path_a": a["image_path"],
        "path_b": b["image_path"],
    }


if __name__ == "__main__":
    main()
