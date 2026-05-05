# Data and Artifact Availability

The manuscript uses public ultrasound datasets from their original providers:

- TN5000 for thyroid nodule ultrasound.
- BUSI for breast ultrasound lesions.
- AUL for annotated liver ultrasound.

This repository does not redistribute dataset images or generated ROI image folders. It includes table-level summaries, prediction CSVs, validation thresholds, fitted temperatures, and provenance files that can be shared without packaging the image datasets.

Restricted or omitted artifacts:

- dataset image files;
- generated ROI image folders;
- trained classifier checkpoints;
- detector weights;
- ONNX model binaries;
- Android APK binaries;
- full raw training logs.

The manuscript and this repository therefore support table regeneration and provenance audit, not full dataset mirroring.
