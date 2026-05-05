# Artifact checklist

This checklist documents the public repository boundary used for the UL-TransXNet artifact.

## Included and auditable

- Model source code for UL-TransXNet and baseline/ablation architectures.
- Training, evaluation, detector, calibration, statistics, and figure-generation scripts.
- Portable YAML protocol summaries for TN5000, BUSI, AUL, ablation, and automatic ROI experiments.
- Compact result CSV files used by the manuscript.
- Manuscript source, bibliography, PDF, and selected final figures.
- Android ONNX Runtime demonstration source and export utilities.
- Smoke test and table reproduction scripts.
- Environment specification and citation metadata.

## Excluded by design

- Raw medical datasets and generated ROI image folders.
- Trained checkpoint weights and YOLO detector weights.
- ONNX binaries and Android APKs.
- Full training logs and intermediate run directories.
- Local cache folders and queue status files.

## Credibility checks

Before a release or submission snapshot, run:

```powershell
python scripts/smoke_test.py --model ul-transxnet --num-classes 2 --input-size 256
python scripts/reproduce_main_tables.py --results-dir results --out-dir reproduced_tables
python scripts/validate_artifact.py
```

A passing snapshot should contain no committed datasets, model weights, ONNX binaries, APKs, or oversized archives.

## Known limitation

Some historical scripts retain the original Windows paths used during the experiments. These paths are provenance hints, not requirements. Use `configs/*.yaml`, environment variables, or command-line arguments to adapt them to a new machine.
