from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "DATA.md",
    "REPRODUCE.md",
    "paper/main.tex",
    "paper/main.pdf",
    "paper/supplementary_material.tex",
    "paper/supplementary_material.pdf",
    "paper/refs.bib",
    "paper/fig_architecture_crop.pdf",
    "paper/fig_roi_robustness_curve_crop.pdf",
    "results/strict_20260514/analysis_labels/analysis_label_snapshot.csv",
    "results/strict_20260514/analysis_labels/analysis_label_manifest_public.json",
    "results/strict_20260514/manuscript_tables/table3_main_benchmark.csv",
    "results/strict_20260514/manuscript_tables/table8_cross_organ.csv",
    "results/strict_20260514/mobile/strict_two_device_mobile_summary_20260514.csv",
]

STALE_PATTERNS = [
    r"MUDD\+DA-noMCA",
    r"TransXNet-GGG",
    r"\bGGG\b",
    r"\bGG-\b",
    r"paper-log",
    r"paperlog",
]

TEXT_EXTENSIONS = {".md", ".tex", ".txt", ".csv", ".json", ".yml", ".yaml", ".cff"}
PATH_LEAK_RE = re.compile(r"C:\\Users\\Afr1ste\\PycharmProjects\\Thyroid", re.IGNORECASE)


def iter_public_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
    return files


def check_required() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"missing required file: {rel}")
    return errors


def check_stale_text() -> list[str]:
    errors: list[str] = []
    patterns = [(p, re.compile(p)) for p in STALE_PATTERNS]
    for path in iter_public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT).as_posix()
        for raw, pattern in patterns:
            if pattern.search(text):
                errors.append(f"stale string {raw!r} in {rel}")
    return errors


def check_path_leaks() -> list[str]:
    errors: list[str] = []
    for path in iter_public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PATH_LEAK_RE.search(text):
            errors.append(f"absolute local path leak in {path.relative_to(ROOT).as_posix()}")
    return errors


def check_label_counts() -> list[str]:
    expected = {
        "TN5000": {0: 1435, 1: 3565},
        "BUSI": {0: 435, 1: 212},
        "AUL": {0: 183, 1: 452},
    }
    label_path = ROOT / "results/strict_20260514/analysis_labels/analysis_label_snapshot.csv"
    if not label_path.exists():
        return ["label snapshot is missing"]

    counts: dict[str, dict[int, int]] = {}
    with label_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = row["dataset"]
            label = int(row["analysis_label"])
            counts.setdefault(dataset, {}).setdefault(label, 0)
            counts[dataset][label] += 1

    errors: list[str] = []
    for dataset, expected_counts in expected.items():
        actual = counts.get(dataset, {})
        if actual != expected_counts:
            errors.append(f"label count mismatch for {dataset}: expected {expected_counts}, got {actual}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_required())
    errors.extend(check_stale_text())
    errors.extend(check_path_leaks())
    errors.extend(check_label_counts())

    if errors:
        print("ERROR: artifact validation failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: artifact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
