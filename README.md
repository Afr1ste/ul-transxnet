# UL-TransXNet

This repository contains the public reproducibility package for the manuscript:

**UL-TransXNet: A Compact ROI Framework for Multi-Dataset Ultrasound Lesion Classification**

UL-TransXNet is a compact ROI-based TransXNet-family classifier for benign--malignant ultrasound lesion classification. The current manuscript positions the evidence conservatively: the model is evaluated under dataset-specific training on TN5000, BUSI, and AUL, and the results support multi-dataset in-domain competitiveness rather than external cross-organ generalization or clinical deployment validation.

## Current Package Scope

Included:

- manuscript LaTeX source and compiled PDF under `paper/`;
- current manuscript figures and figure-generation scripts;
- model source files under `src/models/`;
- training, detector, table-generation, and evaluation scripts under `scripts/`;
- frozen summary CSVs, prediction CSVs, table artifacts, and provenance under `results/`;
- model-selection protocol and source-file checksums in `MODEL_SELECTION_PROTOCOL.md`;
- Android prototype source under `android/`, excluding ONNX binaries, APKs, and build outputs.

Not included:

- dataset images or generated ROI image folders;
- trained checkpoints, detector weights, ONNX exports, or APK binaries;
- large raw training logs and local build artifacts.

These restrictions avoid redistributing third-party dataset images and large model artifacts. The CSV-level package is intended to reproduce manuscript tables and audit the reported result provenance.

## Key Files

- `paper/main.pdf`: current compiled manuscript.
- `paper/main.tex` and `paper/sections/`: manuscript source.
- `MODEL_SELECTION_PROTOCOL.md`: validation-only selection rule, selected row, and source-checksum index.
- `RESULTS_PROVENANCE.md`: table/figure provenance map.
- `results/no_retrain_revision_20260505/`: no-retrain revision tables and model-selection source index.
- `results/high_roi_no_retrain_20260505/`: TN5000 oracle/automatic/full-image probe, localization-robustness probes, case-level diagnostic statistics, and reliability-diagram source bins.
- `results/frozen_source_logs/`: copied source CSVs from frozen completed-result logs.
- `configs/protocol_summary.yaml`: compact dataset/protocol configuration summary.

## No-Retrain Boundary

The 2026-05-05 revision used frozen completed-result CSVs and existing automatic-ROI summaries. It did not reload current intermediate datasets, split manifests, checkpoints, model weights, or image folders because working dataset folders may have label drift relative to the frozen logs.

## Reproduction Entry Points

See `REPRODUCE.md` for table-regeneration notes and expected boundaries. The scripts are provided for transparency, but full end-to-end retraining requires downloading the original datasets from their providers and reconstructing the local dataset layout.
