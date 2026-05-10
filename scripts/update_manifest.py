"""Regenerate MANIFEST.csv for the public repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if ".git/" in rel or rel == "MANIFEST.csv":
        return False
    if "__pycache__" in path.parts:
        return False
    return not any(rel.lower().endswith(ext) for ext in IGNORED_GENERATED_EXTENSIONS)


def main() -> None:
    release_root = ROOT / "results" / "provenance_release_20260510"
    if release_root.exists():
        release_rows = []
        for path in sorted(release_root.rglob("*")):
            if not path.is_file() or path.name == "release_manifest.csv":
                continue
            release_rows.append(
                {
                    "path": path.relative_to(release_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        with (release_root / "release_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
            writer.writeheader()
            writer.writerows(release_rows)
        print(f"Wrote release_manifest.csv with {len(release_rows)} files")

    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or not include(path):
            continue
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (ROOT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote MANIFEST.csv with {len(rows)} files")


if __name__ == "__main__":
    main()
