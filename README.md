# UL-TransXNet

This repository contains the public reproducibility package for the manuscript:

**A Reproducible ROI-Robust Evaluation and Mobile Distillation Framework for Ultrasound Lesion Classification**

This repository contains the CSV-level reproducibility package for an ROI-based ultrasound lesion-classification study. The current manuscript positions the evidence conservatively: TransXNet-family models are used as high-capacity teachers and design-space probes, EfficientFormer students are used for mobile deployment, and TN5000/BUSI/AUL results support protocol-separated public-benchmark evidence rather than external clinical validation.

## Current Package Scope

Included:

- CMPB manuscript LaTeX source, supplementary source, and compiled PDFs under `paper/`;
- current manuscript figures and figure-generation scripts;
- model source files under `src/models/`;
- training, detector, table-generation, and evaluation scripts under `scripts/`;
- frozen summary CSVs, prediction CSVs, label-snapshot manifest, table artifacts, and provenance under `results/`;
- model-selection protocol and source-file checksums in `MODEL_SELECTION_PROTOCOL.md`;
- Android prototype source under `android/`, including the headless batch runner used for the two-device repeat, excluding ONNX binaries, APKs, and build outputs.

Not included:

- dataset images or generated ROI image folders;
- trained checkpoints, detector weights, ONNX exports, or APK binaries;
- large raw training logs and local build artifacts.

These restrictions avoid redistributing third-party dataset images and large model artifacts. The CSV-level package is intended to reproduce manuscript tables and audit the reported result provenance.

## Key Files

- `paper/main.pdf`: current compiled manuscript.
- `paper/supplementary_material.pdf`: current compiled supplementary material.
- `paper/main.tex`, `paper/supplementary_material.tex`, and `paper/refs.bib`: manuscript source.
- `MODEL_SELECTION_PROTOCOL.md`: validation-only selection rule, selected row, and source-checksum index.
- `RESULTS_PROVENANCE.md`: table/figure provenance map.
- `results/provenance_release_20260510/`: public label snapshot, per-case predictions, ROI robustness predictions, BUSI duplicate audit, Android two-device repeat, and release-level SHA256 manifest.
- `results/provenance_release_20260510/predictions/recomputed_paperlog_labels/`: manuscript benchmark tables recomputed from prediction probabilities plus the frozen paper-log labels, including a label-mismatch audit for historical logs.
- `results/no_retrain_revision_20260505/`: no-retrain revision tables and model-selection source index.
- `results/high_roi_no_retrain_20260505/`: TN5000 oracle/automatic/full-image probe, localization-robustness probes, case-level diagnostic statistics, and reliability-diagram source bins.
- `results/frozen_source_logs/`: copied source CSVs from frozen completed-result logs.
- `configs/protocol_summary.yaml`: compact dataset/protocol configuration summary.

## Provenance Boundary

The 2026-05-10 release uses a frozen paper-log analysis-label snapshot. The public case manifest and prediction CSVs are in `results/provenance_release_20260510/`. Manuscript benchmark values are recomputed by `scripts/recompute_paperlog_label_metrics.py`, which overrides historical prediction-file labels with the frozen snapshot and writes an explicit mismatch audit. Dataset images, generated ROI folders, model checkpoints, detector weights, ONNX exports, APKs, and raw training logs are not redistributed. The package supports table regeneration and provenance audit, not full dataset mirroring.

## Reproduction Entry Points

See `REPRODUCE.md` for table-regeneration notes and expected boundaries. The scripts are provided for transparency, but full end-to-end retraining requires downloading the original datasets from their providers and reconstructing the local dataset layout.
