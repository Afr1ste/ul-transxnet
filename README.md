# UL-TransXNet

Repository: https://github.com/Afr1ste/ul-transxnet

UL-TransXNet is a lightweight ROI-based ultrasound lesion classification research artifact evaluated on TN5000, BUSI, and AUL. The repository is organized to support review, result auditing, and independent reproduction subject to the redistribution terms of the original medical datasets.

## Repository contents

- `src/models/`: TransXNet-family architecture definitions and lightweight baseline modules used in the study.
- `src/scripts/`: dataset construction, ROI classification, detector evaluation, ablation, calibration, statistics, and figure-generation scripts.
- `src/tools/`: complexity, latency, and trade-off utilities used for manuscript tables and figures.
- `configs/`: dataset- and experiment-level protocol summaries in portable YAML form.
- `scripts/`: lightweight artifact checks, smoke tests, and table reproduction entrypoints.
- `results/`: compact CSV/TeX/Markdown summaries used by the manuscript.
- `paper/`: LaTeX manuscript source, bibliography, PDF, and selected final figures.
- `android/`: Android ONNX Runtime demo source and ONNX export utilities. Model binaries are intentionally excluded.

## Reproducibility scope

This public repository is intended to make the experimental protocol auditable without redistributing restricted or large artifacts. It includes code, configs, compact result files, and the manuscript source. It does not include public medical image datasets, trained checkpoints, ONNX binaries, APKs, or full training logs.

For exact reproduction, obtain the datasets from their original sources, arrange them according to `DATA.md`, install the environment in `environment.yml`, and follow `REPRODUCE.md`.

## Quick checks

```powershell
conda env create -f environment.yml
conda activate ul-transxnet
python scripts/smoke_test.py --model ul-transxnet --num-classes 2 --input-size 256
python scripts/reproduce_main_tables.py --results-dir results --out-dir reproduced_tables
python scripts/validate_artifact.py
```

The smoke test uses random input only. It verifies that the published model code constructs and runs a forward pass; it does not reproduce trained accuracy.

## Citation

Use `CITATION.cff` for repository citation metadata. Dataset and baseline method citations are listed in the manuscript bibliography.

## License and third-party code

See `LICENSE` and `THIRD_PARTY_NOTICES.md`. Dataset files, trained weights, ONNX exports, and APKs are not redistributed here.
