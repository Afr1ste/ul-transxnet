# Artifact Checklist

Included:

- manuscript source and compiled PDF;
- model architecture source files;
- training, evaluation, table-generation, and export scripts;
- frozen CSV/TEX/JSON/TXT summaries used by the manuscript;
- frozen paper-log label snapshot and public per-case manifest;
- case-level averaged prediction CSVs and ROI-robustness prediction CSVs;
- BUSI duplicate-audit CSVs and two-device Android summary CSVs;
- copied source CSVs used by the validation-only model-selection audit;
- Android prototype source and export scripts.

Excluded:

- dataset images and generated ROI image folders;
- image-containing train/validation/test folders and generated crop folders;
- checkpoints, detector weights, ONNX files, APKs, and build outputs;
- raw training logs and large detector prediction-image folders.

Run `python scripts/validate_artifact.py` before release.
