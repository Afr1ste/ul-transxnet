"""Rebuild lightweight Markdown versions of the compact paper result tables.

The public repository contains compact CSV result summaries rather than full run
folders. This script verifies that the expected CSV files are present and writes
human-readable Markdown tables for audit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED = [
    "paper_complexity_performance_tradeoff.csv",
    "paper_model_complexity_table.csv",
    "tn5000_current_protocol_ablation_summary.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce compact manuscript tables from CSV summaries")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("reproduced_tables"))
    return p.parse_args()


def csv_to_markdown(csv_path: Path) -> str:
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED if not (args.results_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required result files: " + ", ".join(missing))

    generated = []
    for csv_path in sorted(args.results_dir.glob("*.csv")):
        md = csv_to_markdown(csv_path)
        out_path = args.out_dir / (csv_path.stem + ".md")
        out_path.write_text(md, encoding="utf-8")
        generated.append(out_path)

    # Minimal semantic checks for the paper's central claims.
    tradeoff_text = (args.results_dir / "paper_complexity_performance_tradeoff.csv").read_text(encoding="utf-8-sig")
    ablation_text = (args.results_dir / "tn5000_current_protocol_ablation_summary.csv").read_text(encoding="utf-8-sig")
    required_tokens = ["UL-TransXNet", "ConvNeXt-Tiny", "Swin-T", "TransXNet-GG", "TransXNet-GGG-MCA"]
    missing_tokens = [tok for tok in required_tokens if tok not in tradeoff_text + ablation_text]
    if missing_tokens:
        raise AssertionError("Expected tokens not found in compact results: " + ", ".join(missing_tokens))

    print("Generated Markdown tables:")
    for path in generated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
