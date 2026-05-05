# Artifact Checklist

Included:

- manuscript source and compiled PDF;
- model architecture source files;
- training, evaluation, table-generation, and export scripts;
- frozen CSV/TEX/JSON/TXT summaries used by the manuscript;
- copied source CSVs used by the validation-only model-selection audit;
- Android prototype source and export scripts.

Excluded:

- dataset images and generated ROI image folders;
- train/validation/test split files that may expose local dataset layouts;
- checkpoints, detector weights, ONNX files, APKs, and build outputs;
- raw training logs and large detector prediction-image folders.

Run `python scripts/validate_artifact.py` before release.
