# Data Availability

The study uses public ultrasound datasets, but image files are not redistributed in this repository.

## Public Datasets

- TN5000: thyroid ultrasound lesion classification.
- BUSI: breast ultrasound benign/malignant lesion classification.
- AUL: liver ultrasound benign/malignant lesion classification.

Users should obtain the datasets from their original providers and comply with the corresponding licenses and citation requirements.

## Split and Protocol Metadata

The current strict analyses use fixed split/protocol metadata distributed under `results/strict_20260514/`.

This CSV contains only:

- dataset name
- case identifier
- label
- label name
- split
- observation counts

It does not include image pixels, absolute local image paths, or private source-log path lists.

## Excluded Artifacts

The following artifacts are intentionally not committed:

- original medical images
- ROI image folders
- trained checkpoints
- detector weights
- ONNX binaries
- Android APKs
- full training logs

Compact CSV/JSON summaries needed for manuscript-level audit are included under `results/strict_20260514/`.
