"""Validate the public UL-TransXNet reproducibility package.

This script intentionally checks only repository artifacts. It does not load
image folders, checkpoints, ONNX exports, APKs, or current local dataset
manifests.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".tflite",
    ".apk",
    ".aab",
    ".zip",
    ".pyc",
}

IGNORED_GENERATED_EXTENSIONS = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".spl",
    ".synctex.gz",
}

REQUIRED_PATHS = [
    "README.md",
    "REPRODUCE.md",
    "DATA.md",
    "MANIFEST.csv",
    "MODEL_SELECTION_PROTOCOL.md",
    "RESULTS_PROVENANCE.md",
    "paper/main.pdf",
    "paper/main.tex",
    "paper/supplementary_material.pdf",
    "paper/supplementary_material.tex",
    "paper/figures/fig_reliability_diagram_case_level.png",
    "paper/figures/fig_roi_robustness_curve_crop.pdf",
    "paper/figures/fig_mobile_latency_tradeoff_crop.pdf",
    "results/high_roi_no_retrain_20260505/case_level_diagnostic_statistics.csv",
    "results/high_roi_no_retrain_20260505/tn5000_oracle_auto_full_probe.csv",
    "results/high_roi_no_retrain_20260505/tn5000_localization_robustness_probe.csv",
    "results/provenance_release_20260510/README.md",
    "results/provenance_release_20260510/release_manifest.csv",
    "results/provenance_release_20260510/label_snapshot/paper_log_case_manifest_public.csv",
    "results/provenance_release_20260510/predictions/paperlog_per_case_averaged_predictions_public.csv",
    "results/provenance_release_20260510/predictions/roi_robustness/box_noise_robustness_metrics.csv",
    "results/provenance_release_20260510/android_two_device/README.md",
    "results/no_retrain_revision_20260505/model_selection_source_files.csv",
    "results/no_retrain_revision_20260505/validation_selection_audit.csv",
    "results/frozen_source_logs",
    "src/models/transxnetggg.py",
    "scripts/build_provenance_release.py",
    "src/scripts/generate_high_roi_no_retrain_tables.py",
    "src/scripts/generate_no_retrain_revision_tables.py",
    "src/scripts/audit_busi_duplicates.py",
    "android/README.md",
    "android/app/src/main/java/com/afriste/tn5000ortdemo/HeadlessBatchActivity.kt",
    "android/app/src/main/java/com/afriste/tn5000ortdemo/LabelSnapshot.kt",
    "android/app/src/main/assets/PLACE_ONNX_MODELS_HERE.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_repo_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        and not any(str(p).lower().endswith(ext) for ext in IGNORED_GENERATED_EXTENSIONS)
    ]


def check_required_paths() -> None:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))


def check_forbidden_files(files: list[Path]) -> None:
    forbidden = [
        str(p.relative_to(ROOT))
        for p in files
        if p.suffix.lower() in FORBIDDEN_EXTENSIONS
    ]
    if forbidden:
        raise SystemExit("Forbidden binary/build artifacts found:\n" + "\n".join(forbidden))


def check_large_files(files: list[Path], max_mb: float = 25.0) -> None:
    max_bytes = int(max_mb * 1024 * 1024)
    large = [
        f"{p.relative_to(ROOT)},{p.stat().st_size}"
        for p in files
        if p.stat().st_size > max_bytes
    ]
    if large:
        raise SystemExit("Files larger than limit:\n" + "\n".join(large))


def check_model_selection_hashes() -> None:
    index_path = ROOT / "results/no_retrain_revision_20260505/model_selection_source_files.csv"
    with index_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    problems: list[str] = []
    for row in rows:
        rel_value = row.get("path") or row.get("Path")
        sha_value = row.get("sha256") or row.get("SHA256")
        if rel_value is None or sha_value is None:
            raise SystemExit("model_selection_source_files.csv must include Path and SHA256 columns")
        rel = rel_value.replace("\\", "/")
        copied = ROOT / "results/frozen_source_logs" / rel
        if not copied.exists():
            problems.append(f"missing copied source CSV: {rel}")
            continue
        expected = sha_value.strip().lower()
        actual = sha256(copied)
        if actual != expected:
            problems.append(f"sha256 mismatch: {rel} expected={expected} actual={actual}")

    if problems:
        raise SystemExit("Frozen source CSV validation failed:\n" + "\n".join(problems))


def check_manifest(files: list[Path]) -> None:
    manifest_path = ROOT / "MANIFEST.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}

    expected_files = {
        p.relative_to(ROOT).as_posix(): p
        for p in files
        if p.relative_to(ROOT).as_posix() != "MANIFEST.csv"
    }

    problems: list[str] = []
    missing = sorted(set(expected_files) - set(rows))
    extra = sorted(set(rows) - set(expected_files))
    if missing:
        problems.append("missing manifest rows: " + ", ".join(missing[:10]))
    if extra:
        problems.append("extra manifest rows: " + ", ".join(extra[:10]))

    for rel, path in expected_files.items():
        row = rows.get(rel)
        if row is None:
            continue
        if int(row["size_bytes"]) != path.stat().st_size:
            problems.append(f"size mismatch: {rel}")
        if row["sha256"].strip().lower() != sha256(path):
            problems.append(f"sha256 mismatch: {rel}")

    if problems:
        raise SystemExit("MANIFEST.csv validation failed:\n" + "\n".join(problems))


def check_release_manifest() -> None:
    release_root = ROOT / "results/provenance_release_20260510"
    manifest_path = release_root / "release_manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}

    expected = {
        p.relative_to(release_root).as_posix(): p
        for p in release_root.rglob("*")
        if p.is_file() and p.name != "release_manifest.csv"
    }

    problems: list[str] = []
    missing = sorted(set(expected) - set(rows))
    extra = sorted(set(rows) - set(expected))
    if missing:
        problems.append("missing release-manifest rows: " + ", ".join(missing[:10]))
    if extra:
        problems.append("extra release-manifest rows: " + ", ".join(extra[:10]))

    for rel, path in expected.items():
        row = rows.get(rel)
        if row is None:
            continue
        if int(row["size_bytes"]) != path.stat().st_size:
            problems.append(f"release size mismatch: {rel}")
        if row["sha256"].strip().lower() != sha256(path):
            problems.append(f"release sha256 mismatch: {rel}")

    if problems:
        raise SystemExit("release_manifest.csv validation failed:\n" + "\n".join(problems))


def main() -> None:
    files = iter_repo_files()
    check_required_paths()
    check_forbidden_files(files)
    check_large_files(files)
    check_model_selection_hashes()
    check_release_manifest()
    check_manifest(files)
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"OK: {len(files)} files, {total_bytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
