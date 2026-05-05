# Model and artifact availability

This repository publishes architecture definitions, training/evaluation scripts, compact result summaries, and Android demo source. It intentionally does not redistribute trained model weights or exported binaries.

## Included

- UL-TransXNet / TransXNet-family model definitions in `src/models/`.
- Baseline and ablation training scripts in `src/scripts/`.
- ONNX export utilities in `android/export/`.
- Android ONNX Runtime demo source in `android/TN5000OrtDemoComplete/`.

## Not included

- `.pth` / `.pt` training checkpoints.
- `.onnx` model exports.
- Android `.apk` files.
- Full training logs and intermediate run folders.

## Rationale

The excluded files are large, environment-specific, or coupled to dataset redistribution constraints. The published scripts and configs are sufficient to rebuild these artifacts after obtaining the datasets and installing the environment.

## Rebuilding model binaries

After training, use the ONNX export scripts under `android/export/`. Place exported ONNX files in the Android app asset folder locally; do not commit them unless redistribution permissions are explicitly resolved.
