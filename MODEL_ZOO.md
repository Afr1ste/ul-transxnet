# Model Artifacts

This public repository does not include trained checkpoints, detector weights,
ONNX exports, or APK binaries.

The omitted artifacts are large and/or tied to third-party dataset files that
cannot be redistributed here. The repository instead provides:

- model source code under `src/models/`;
- training and evaluation entry points under `src/scripts/`;
- Android ONNX Runtime demo source under `android/`;
- frozen CSV outputs and SHA256 provenance under `results/`.

For the 2026-05-05 no-retrain revision, reported values are audited through
`MODEL_SELECTION_PROTOCOL.md` and
`results/no_retrain_revision_20260505/model_selection_source_files.csv`.
