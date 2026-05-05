"""Validate public artifact hygiene before pushing or archiving.

This script checks for accidentally committed large binaries, datasets, model
weights, ONNX/APK files, and key reproducibility metadata. It is intentionally
conservative and fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".onnx", ".apk", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".nii", ".dcm", ".dicom",
}
REQUIRED_FILES = [
    "README.md",
    "DATA.md",
    "REPRODUCE.md",
    "MODEL_ZOO.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "environment.yml",
    "configs/common.yaml",
    "scripts/smoke_test.py",
    "scripts/reproduce_main_tables.py",
    "results/paper_complexity_performance_tradeoff.csv",
    "results/tn5000_current_protocol_ablation_summary.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate public artifact hygiene")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--max-file-mb", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    problems: list[str] = []

    for rel in REQUIRED_FILES:
        if not (repo / rel).exists():
            problems.append(f"missing required file: {rel}")

    for path in repo.rglob("*"):
        if ".git" in path.parts:
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        suffix = path.suffix.lower()
        size_mb = path.stat().st_size / (1024 * 1024)
        if suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact suffix: {rel}")
        if size_mb > args.max_file_mb:
            problems.append(f"large file over {args.max_file_mb:.1f} MB: {rel} ({size_mb:.1f} MB)")

    if problems:
        print("Artifact validation failed:")
        for item in problems:
            print(f"- {item}")
        return 1

    print("Artifact validation passed.")
    print(f"Checked repository: {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
