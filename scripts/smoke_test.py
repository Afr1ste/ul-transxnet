"""Minimal public-package smoke test."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_validator() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_artifact.py")],
        check=True,
        cwd=ROOT,
    )


def check_selection_table() -> None:
    path = ROOT / "results/no_retrain_revision_20260505/validation_selection_audit.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r.get("selected", "").lower() in {"yes", "true", "1"}]
    if len(rows) < 4:
        raise SystemExit(f"Expected at least 4 model-family rows, found {len(rows)}")
    if len(selected) != 1:
        raise SystemExit(f"Expected exactly one selected row, found {len(selected)}")
    print(f"Selection audit rows: {len(rows)}; selected: {selected[0].get('variant')}")


def main() -> None:
    run_validator()
    check_selection_table()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
