"""List the checked-in table artifacts used by the manuscript.

The no-retrain public package includes frozen CSV/TEX outputs rather than raw
datasets or checkpoints. This helper verifies that the table artifacts are
present and prints their locations for manual inspection.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TABLE_GROUPS = {
    "no_retrain_revision": ROOT / "results/no_retrain_revision_20260505",
    "main_result_statistics": ROOT / "results/main_result_statistics_20260421_045323",
    "offline_stat_tests": ROOT / "results/paper_offline_stat_tests_20260504_191531",
    "calibration": ROOT / "results/paper_statistical_calibration_bboxfix_20260504_182514",
    "high_roi_no_retrain": ROOT / "results/high_roi_no_retrain_20260505",
    "tn5000_auto_roi": ROOT / "results/tn5000_auto_roi_final_summary_20260503_161324",
    "busi_aul_auto_roi": ROOT / "results/busi_aul_closed_loop_auto_roi_bboxfix_20260504_182516",
    "android": ROOT / "results/android_paper_batch_20260501_162525",
    "provenance_release": ROOT / "results/provenance_release_20260510",
    "recomputed_paperlog_labels": ROOT
    / "results/provenance_release_20260510/predictions/recomputed_paperlog_labels",
}


def main() -> None:
    missing = [name for name, path in TABLE_GROUPS.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing table groups: " + ", ".join(missing))

    for name, path in TABLE_GROUPS.items():
        artifacts = sorted(
            p.relative_to(ROOT).as_posix()
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".tex", ".json", ".md", ".txt"}
        )
        print(f"[{name}] {len(artifacts)} artifacts")
        for rel in artifacts:
            print(f"  {rel}")


if __name__ == "__main__":
    main()
