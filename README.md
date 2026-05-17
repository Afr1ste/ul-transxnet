# UltraDrill Edge Ultrasound Classification

This repository is the clean public artifact for the CMPB-oriented manuscript:

**UltraDrill: A Distillation and Optimization Framework for Edge Ultrasound Lesion Classification**

The repository was rebuilt from an empty public branch on 2026-05-14 to remove stale internal experiment names and mixed-label artifacts. It contains the current manuscript package, strict result summaries, split/protocol metadata, and lightweight validation scripts.

## What Is Included

- `paper/`: current LaTeX manuscript, supplementary material, compiled PDFs, submission notes, declaration draft, and final figures.
- `results/strict_20260514/`: compact public CSV/JSON evidence used by the current manuscript.
- `tools/make_clean_submission_figures.py`: script used to regenerate the cleaned architecture and ROI robustness figures.
- `scripts/validate_artifact.py`: repository integrity and stale-string checks.
- `scripts/reproduce_main_tables.py`: copies manuscript table CSVs to a review output folder for quick audit.
- `scripts/build_cmpb_submission_package.py`: builds a flat source/PDF/result package for review or Editorial Manager upload.

## What Is Not Included

This repository does not redistribute medical image datasets, trained checkpoints, ONNX binaries, APKs, or full training logs. Those files are intentionally excluded because of size and dataset/license constraints.

The public result files are designed for auditability. They preserve table-level evidence and protocol metadata, while removing absolute local paths and private source-log path lists.

## Quick Check

```powershell
python scripts/validate_artifact.py
python scripts/reproduce_main_tables.py --out-dir reproduced_tables
python scripts/build_cmpb_submission_package.py
```

Expected validation result:

```text
OK: artifact validation passed
```

## Current Evidence Snapshot

- Split/protocol metadata: see `results/strict_20260514/`.
- Metadata source SHA-256: see `results/strict_20260514/`.
- Main table CSVs: `results/strict_20260514/manuscript_tables/`
- Teacher-student ROI stress summary: `results/strict_20260514/teacher_student_roi_robustness/teacher_student_roi_drop_summary.csv`
- Teacher-side ROI robustness CSV: `results/strict_20260514/roi_robustness/roi_robustness_metrics_public.csv`
- Two-device Android summary: `results/strict_20260514/mobile/strict_two_device_mobile_summary_20260514.csv`
- Cross-organ summary: `results/strict_20260514/cross_organ/aggregate_metrics_by_domain.csv`

## Scope

This is a review and reproducibility package, not a dataset mirror. The manuscript makes bounded claims around UltraDrill: localization-stress-guided teacher--student optimization, fixed edge post-processing and verification, two-device Android feasibility, and limits of zero-shot cross-organ transfer.
