from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "strict_20260514" / "manuscript_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy manuscript table CSVs to an output directory.")
    parser.add_argument("--out-dir", default="reproduced_tables", help="Output directory for copied CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(TABLE_DIR.glob("*.csv")):
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst.relative_to(ROOT).as_posix())

    for path in copied:
        print(path)
    print(f"copied {len(copied)} table files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
