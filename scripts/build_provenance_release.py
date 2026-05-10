"""Build the public provenance-release artifacts from the local experiment tree.

The generated directory contains CSV-level artifacts only. It intentionally
does not copy image files, ROI crops, checkpoints, detector weights, ONNX
exports, APKs, or raw training logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


RELEASE_NAME = "provenance_release_20260510"
USER_ROOT = r"[A-Za-z]:\\Users\\[^\\]+"
USER_ROOT_ESC = r"[A-Za-z]:\\\\Users\\\\[^\\]+"
PROJECTS = "Pycharm" + "Projects"
PRIVATE_WORKSPACE = "Thy" + "roid"
PUBLIC_WORKSPACE = "ul-transxnet_public_reset_20260505"
ONE_DRIVE = "One" + "Drive"
NOTES_DIR = "My" + " Notes"
MANUSCRIPT_ROOT = "pr_ultrasound_lesion_classification"
SENSITIVE_PATTERNS = [
    (re.compile(rf"{USER_ROOT}\\{PROJECTS}\\{PRIVATE_WORKSPACE}"), "<LOCAL_THYROID_ROOT>"),
    (re.compile(rf"{USER_ROOT_ESC}\\\\{PROJECTS}\\\\{PRIVATE_WORKSPACE}"), "<LOCAL_THYROID_ROOT>"),
    (re.compile(rf"[A-Za-z]:/Users/[^/]+/{PROJECTS}/{PRIVATE_WORKSPACE}"), "<LOCAL_THYROID_ROOT>"),
    (
        re.compile(rf"{USER_ROOT}\\{PROJECTS}\\{PUBLIC_WORKSPACE}"),
        "<PUBLIC_REPO_ROOT>",
    ),
    (
        re.compile(rf"{USER_ROOT_ESC}\\\\{PROJECTS}\\\\{PUBLIC_WORKSPACE}"),
        "<PUBLIC_REPO_ROOT>",
    ),
    (
        re.compile(
            rf"{USER_ROOT}\\{ONE_DRIVE}\\{NOTES_DIR}\\tex\\{MANUSCRIPT_ROOT}(?:\\cmpb_submission_20260510)?"
        ),
        "<LOCAL_MANUSCRIPT_ROOT>",
    ),
    (
        re.compile(
            rf"{USER_ROOT_ESC}\\\\{ONE_DRIVE}\\\\{NOTES_DIR}\\\\tex\\\\{MANUSCRIPT_ROOT}(?:\\\\cmpb_submission_20260510)?"
        ),
        "<LOCAL_MANUSCRIPT_ROOT>",
    ),
    (re.compile(r"[A-Za-z]:\\Browsers'Downloads"), "<LOCAL_DOWNLOADS>"),
    (re.compile(r"[A-Za-z]:\\\\Browsers'Downloads"), "<LOCAL_DOWNLOADS>"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\]+\\anaconda3"), "<LOCAL_CONDA_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\]+\\\\anaconda3"), "<LOCAL_CONDA_ROOT>"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def sanitize_text(text: str) -> str:
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(sanitize_text(src.read_text(encoding="utf-8-sig")), encoding="utf-8")


def rel_image_path(dataset: str, image_id: str) -> str:
    ext = ".jpg" if dataset == "TN5000" or dataset == "AUL" else ".png"
    return f"{dataset}/JPEGImages/{image_id}{ext}"


def rel_annotation_path(dataset: str, image_id: str) -> str:
    return f"{dataset}/Annotations/{image_id}.xml"


def find_xml(thyroid_root: Path, dataset: str, image_id: str) -> Path | None:
    if dataset == "TN5000":
        path = thyroid_root / "TN5000_forReview" / "Annotations" / f"{image_id}.xml"
    elif dataset == "BUSI":
        path = thyroid_root / "busi" / "busi_voc_v3_square_consistent" / "Annotations" / f"{image_id}.xml"
    elif dataset == "AUL":
        path = thyroid_root / "aul" / "aul_voc_roi_v1" / "Annotations" / f"{image_id}.xml"
    else:
        return None
    return path if path.is_file() else None


def parse_first_box(path: Path | None) -> dict[str, object]:
    out: dict[str, object] = {
        "bbox_count": 0,
        "bbox_xmin": "",
        "bbox_ymin": "",
        "bbox_xmax": "",
        "bbox_ymax": "",
    }
    if path is None:
        return out
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return out
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            coords = {
                "bbox_xmin": int(float(box.findtext("xmin", ""))),
                "bbox_ymin": int(float(box.findtext("ymin", ""))),
                "bbox_xmax": int(float(box.findtext("xmax", ""))),
                "bbox_ymax": int(float(box.findtext("ymax", ""))),
            }
        except ValueError:
            continue
        boxes.append(coords)
    out["bbox_count"] = len(boxes)
    if boxes:
        out.update(boxes[0])
    return out


def sanitize_path_value(value: str) -> str:
    value = (value or "").replace("\\", "/")
    markers = [
        "/TN5000_forReview/",
        "/busi_voc_v3_square_consistent/",
        "/aul_voc_roi_v1/",
    ]
    for marker in markers:
        if marker in value:
            return value.split(marker, 1)[1]
    return value


def build_label_manifest(thyroid_root: Path, out_dir: Path) -> None:
    src_dir = thyroid_root / "eval_reports" / "paper_log_label_reconstruction_20260505"
    rows = read_csv(src_dir / "paper_log_case_labels.csv")
    out_rows: list[dict[str, object]] = []
    for row in rows:
        dataset = row["dataset"].strip().upper()
        image_id = row["image_id"].strip()
        xml = find_xml(thyroid_root, dataset, image_id)
        out = {
            "dataset": dataset,
            "image_id": image_id,
            "split": row.get("splits_seen", ""),
            "true_label": row.get("true_label", ""),
            "label_name": row.get("label_name", ""),
            "image_relpath": rel_image_path(dataset, image_id),
            "annotation_relpath": rel_annotation_path(dataset, image_id),
            "n_observations": row.get("n_observations", ""),
            "source_count": row.get("source_count", ""),
            "source_methods_seen": row.get("source_methods_seen", ""),
            "source_paths": row.get("source_paths", ""),
        }
        out.update(parse_first_box(xml))
        out_rows.append(out)
    fields = [
        "dataset",
        "image_id",
        "split",
        "true_label",
        "label_name",
        "image_relpath",
        "annotation_relpath",
        "bbox_count",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "n_observations",
        "source_count",
        "source_methods_seen",
        "source_paths",
    ]
    label_dir = out_dir / "label_snapshot"
    write_csv(label_dir / "paper_log_case_manifest_public.csv", out_rows, fields)

    for name in [
        "paper_log_label_summary.csv",
        "paper_log_label_sources.csv",
        "paper_log_label_conflicts.csv",
        "paper_log_count_lines_from_logs.csv",
        "README_paper_log_labels.md",
    ]:
        copy_text(src_dir / name, label_dir / name)

    summary = {
        "release": RELEASE_NAME,
        "source_snapshot": "eval_reports/paper_log_label_reconstruction_20260505",
        "case_count": len(out_rows),
        "counts_by_dataset_label": {},
        "notes": [
            "Labels come from the frozen paper-log snapshot.",
            "Bounding boxes are read from local VOC XML files for coordinate provenance only.",
            "No image files are included in this release.",
        ],
    }
    counts: dict[str, dict[str, int]] = {}
    for row in out_rows:
        counts.setdefault(str(row["dataset"]), {}).setdefault(str(row["true_label"]), 0)
        counts[str(row["dataset"])][str(row["true_label"])] += 1
    summary["counts_by_dataset_label"] = counts
    (label_dir / "paper_log_case_manifest_public.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sanitize_prediction_csv(src: Path, dst: Path, dataset_hint: str | None = None) -> None:
    rows = read_csv(src)
    out_rows: list[dict[str, object]] = []
    fieldnames: list[str] | None = None
    for row in rows:
        row = dict(row)
        dataset = (row.get("dataset") or dataset_hint or "TN5000").upper()
        image_id = row.get("image_id") or row.get("base_name") or ""
        if "image_path" in row:
            row["image_relpath"] = sanitize_path_value(row.pop("image_path"))
        elif image_id:
            row["image_relpath"] = rel_image_path(dataset, image_id)
        if "representative_image_path_at_evaluation" in row:
            row["representative_image_relpath"] = sanitize_path_value(
                row.pop("representative_image_path_at_evaluation")
            )
        out_rows.append(row)
        if fieldnames is None:
            fieldnames = list(row.keys())
        else:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    write_csv(dst, out_rows, fieldnames or [])


def build_prediction_artifacts(thyroid_root: Path, out_dir: Path) -> None:
    pred_dir = out_dir / "predictions"
    offline = thyroid_root / "eval_reports" / "paperlog_remaining_p1p2_20260507_223840" / "offline_artifacts"
    for name in [
        "paperlog_per_case_averaged_predictions.csv",
        "paperlog_per_case_metric_summary.csv",
        "busi_aul_fixed_specificity_operating_points.csv",
        "busi_aul_roc_operating_curve_points.csv",
        "method_module_detail_table.csv",
        "summary.md",
    ]:
        src = offline / name
        dst = pred_dir / ("paperlog_per_case_averaged_predictions_public.csv" if name.endswith("predictions.csv") else name)
        if name.endswith("predictions.csv"):
            sanitize_prediction_csv(src, dst)
        else:
            copy_text(src, dst)

    supplement = thyroid_root / "eval_reports" / "paperlog_p1p2_supplement_paperlabels_20260510"
    for name in [
        "p1_case_bootstrap_ci.csv",
        "p1_run_level_metrics.csv",
        "p2_oracle_threshold_transfer_aggregate.csv",
        "p2_oracle_threshold_transfer_per_seed.csv",
        "summary.md",
    ]:
        copy_text(supplement / name, pred_dir / "supplement" / name)

    robustness = thyroid_root / "eval_reports" / "paperlog_remaining_p1p2_20260507_223840" / "box_noise_robustness"
    copy_text(robustness / "box_noise_robustness_metrics.csv", pred_dir / "roi_robustness" / "box_noise_robustness_metrics.csv")
    copy_text(robustness / "run_config.json", pred_dir / "roi_robustness" / "run_config.json")
    for src in sorted((robustness / "predictions").glob("*/*/test_predictions.csv")):
        rel = src.relative_to(robustness / "predictions")
        dst = pred_dir / "roi_robustness" / rel.parent / "test_predictions_public.csv"
        sanitize_prediction_csv(src, dst, dataset_hint="TN5000")


def build_busi_audit(thyroid_root: Path, out_dir: Path) -> None:
    audit_dir = out_dir / "busi_duplicate_audit"
    for tag in ["t2", "t5"]:
        src_dir = thyroid_root / "eval_reports" / f"busi_duplicate_audit_20260510_manifestlabels_{tag}"
        dst_dir = audit_dir / f"manifestlabels_{tag}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*"):
            if src.suffix.lower() != ".csv":
                copy_text(src, dst_dir / src.name)
                continue
            rows = read_csv(src)
            out_rows = []
            fieldnames: list[str] = []
            for row in rows:
                clean = {}
                for key, value in row.items():
                    if key in {"image_path", "path_a", "path_b"}:
                        clean[key.replace("path", "relpath")] = sanitize_path_value(value)
                    else:
                        clean[key] = value
                out_rows.append(clean)
                for key in clean:
                    if key not in fieldnames:
                        fieldnames.append(key)
            write_csv(dst_dir / src.name, out_rows, fieldnames)


def build_android_artifacts(thyroid_root: Path, out_dir: Path) -> None:
    android_dir = out_dir / "android_two_device"
    roots = {
        "xiaomi_24129pn74c_20260510_150734": thyroid_root
        / "eval_reports"
        / "android_headless_batches"
        / "headless_file_batch_20260510_150734_paperlog_labels"
        / "headless_file_batch_20260510_150734",
        "samsung_sm_x800_20260510_174443": thyroid_root
        / "eval_reports"
        / "android_headless_batches"
        / "samsung_sm_x800_paperlog_labels_20260510_174443"
        / "headless_file_batch_20260510_174443",
    }
    for name, src_dir in roots.items():
        dst_dir = android_dir / name
        for fname in [
            "aggregate_by_mode.csv",
            "aggregate_by_mode_phase.csv",
            "batch_runs_manifest.csv",
            "batch_summary.txt",
            "device_info.txt",
        ]:
            copy_text(src_dir / fname, dst_dir / fname)
    readme = """# Android two-device repeat

Both device runs use the frozen paper-log TN5000 analysis-label snapshot, 30%
expanded non-square ROI crops, and five hot runs per exported model.

- Xiaomi 24129PN74C: `xiaomi_24129pn74c_20260510_150734`
- Samsung SM-X800: `samsung_sm_x800_20260510_174443`

The exported AUC values match across devices; absolute latency differs by
hardware. These artifacts support mobile feasibility, not clinical deployment.
"""
    (android_dir / "README.md").write_text(readme, encoding="utf-8")


def write_release_manifest(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "release_manifest.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(out_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(out_dir / "release_manifest.csv", rows, ["path", "size_bytes", "sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thyroid-root", type=Path, default=Path(__file__).resolve().parents[2] / "Thyroid")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    thyroid_root = args.thyroid_root.resolve()
    repo_root = args.repo_root.resolve()
    if not thyroid_root.exists():
        raise SystemExit(f"Missing Thyroid root: {thyroid_root}")

    out_dir = repo_root / "results" / RELEASE_NAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    build_label_manifest(thyroid_root, out_dir)
    build_prediction_artifacts(thyroid_root, out_dir)
    build_busi_audit(thyroid_root, out_dir)
    build_android_artifacts(thyroid_root, out_dir)

    readme = """# Provenance release 2026-05-10

This directory contains the public, CSV-level provenance package for the CMPB
submission draft. It includes the frozen paper-log label snapshot, public case
manifest, per-case averaged predictions, ROI robustness prediction CSVs,
BUSI duplicate-audit tables, two-device Android summaries, and SHA256 hashes.

No dataset images, generated ROI image folders, checkpoints, detector weights,
ONNX binaries, APKs, or raw training logs are included.

Important boundaries:

- `label_snapshot/paper_log_case_manifest_public.csv` is the authoritative
  label/split snapshot for the added analyses.
- `predictions/paperlog_per_case_averaged_predictions_public.csv` is the main
  per-case averaged prediction table used for case-level statistics.
- `predictions/roi_robustness/` contains TN5000 test prediction CSVs for the
  GT/noisy/detector ROI robustness benchmark.
- `android_two_device/` contains Xiaomi and Samsung hot-run summaries.
- `release_manifest.csv` records size and SHA256 for every file in this
  release directory.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    write_release_manifest(out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
