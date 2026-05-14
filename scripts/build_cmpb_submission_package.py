from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

SOURCE_FILES = [
    "main.tex",
    "supplementary_material.tex",
    "refs.bib",
    "main.pdf",
    "supplementary_material.pdf",
    "highlights_cmpb.txt",
    "cover_letter_cmpb.md",
    "title_page_cmpb.md",
    "declaration_of_competing_interest.docx",
    "submission_checklist_cmpb.md",
    "artifact_manifest_requirements_cmpb.md",
]

FIGURE_FILES = [
    "fig_roi_preprocessing.png",
    "fig_architecture_crop.pdf",
    "fig_architecture_crop.png",
    "fig_architecture_crop.svg",
    "fig_auto_roi_workflow_crop.pdf",
    "fig_roi_robustness_curve_crop.pdf",
    "fig_mobile_tradeoff_crop.pdf",
    "fig_mobile_tradeoff_crop.png",
]

REPO_FILES = [
    "README.md",
    "DATA.md",
    "REPRODUCE.md",
    "requirements.txt",
    "scripts/validate_artifact.py",
    "scripts/reproduce_main_tables.py",
    "scripts/build_cmpb_submission_package.py",
    "tools/make_clean_submission_figures.py",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_inventory(package_dir: Path) -> None:
    rows = []
    for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(package_dir).as_posix()
        if rel == "FILE_INVENTORY.csv":
            continue
        rows.append({"relative_path": rel, "bytes": path.stat().st_size})
    with (package_dir / "FILE_INVENTORY.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)


def zip_dir(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(package_dir.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flat CMPB upload/review package.")
    parser.add_argument("--out-root", default=str(ROOT / "build"), help="Output root directory.")
    parser.add_argument("--name", default="", help="Optional package directory name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    name = args.name or f"cmpb_submission_package_{datetime.now():%Y%m%d_%H%M%S}"
    package_dir = out_root / name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    for file_name in SOURCE_FILES + FIGURE_FILES:
        copy_file(PAPER / file_name, package_dir / file_name)

    for rel in REPO_FILES:
        copy_file(ROOT / rel, package_dir / rel)

    copy_file(PAPER / "LLM_REVIEW_BRIEF.md", package_dir / "START_HERE_FOR_LLM_REVIEW.md")
    shutil.copytree(ROOT / "results" / "strict_20260514", package_dir / "results" / "strict_20260514")

    note = package_dir / "UPLOAD_TODO.txt"
    note.write_text(
        "Before Editorial Manager upload:\n"
        "1. Fill the corresponding author's telephone number in title_page_cmpb.md and cover_letter_cmpb.md.\n"
        "2. Review the included declaration_of_competing_interest.docx; replace it with the official Elsevier-generated declaration if Editorial Manager requires that exact output.\n"
        "3. Confirm author order, affiliations, and repository URL.\n",
        encoding="utf-8",
    )

    write_inventory(package_dir)
    zip_path = package_dir.with_suffix(".zip")
    zip_dir(package_dir, zip_path)

    print(package_dir)
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
