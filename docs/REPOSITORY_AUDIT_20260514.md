# Repository Audit - 2026-05-14

## Purpose

The public repository was rebuilt after strict-label reruns and manuscript cleanup. The goal is to avoid publishing stale mixed-label outputs, old internal model names, or local-only paths as if they were stable paper evidence.

## Actions

- Started from the cleared `main` branch.
- Added the current CMPB-oriented manuscript source and PDFs.
- Added the current supplementary material.
- Added compact strict result summaries under `results/strict_20260514/`.
- Added a path-sanitized analysis-label snapshot.
- Added public manuscript table CSVs.
- Added validation and table-copy scripts.
- Excluded datasets, checkpoints, ONNX binaries, APKs, and full logs.

## Validation

Run:

```powershell
python scripts/validate_artifact.py
```

The validator checks required files, stale public strings, and absolute local path leakage in public CSV/JSON/Markdown/Tex files.
