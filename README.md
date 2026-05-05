# UL-TransXNet

This repository contains the source code, paper assets, and lightweight reproduction scripts for UL-TransXNet, a lightweight ROI-based ultrasound lesion classification framework evaluated on TN5000, BUSI, and AUL.

## What is included

- `src/models/`: TransXNet-family model definitions and related lightweight baseline modules.
- `src/scripts/`: dataset construction, ROI classification, detector evaluation, ablation, calibration, and figure-generation scripts.
- `src/tools/`: complexity, latency, and trade-off utilities used for paper tables and figures.
- `paper/`: LaTeX manuscript source and selected final figures.
- `results/`: compact CSV/TeX/Markdown summaries used by the manuscript.
- `android/`: Android ONNX Runtime demo source and ONNX export utilities. Model binaries are intentionally excluded.

## What is not included

This repository does not redistribute public medical image datasets, trained checkpoint weights, ONNX binaries, APKs, or intermediate training logs. See `DATA.md` and `MODEL_ZOO.md`.

## Status

This is a research artifact prepared for reproducibility and review. Paths in the original experiment scripts may need to be adjusted to your local dataset locations.
