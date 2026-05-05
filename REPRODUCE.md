# Reproduction guide

This document describes the intended reproduction path for the public artifact. It separates light artifact checks from full training, because full training requires the original datasets and GPU time.

## 1. Environment

Create an environment from `environment.yml` or manually install equivalent versions listed there. The locally tested environment used PyTorch 2.7.1 with CUDA 11.8, timm 1.0.25, MMCV 2.2.0, and Ultralytics 8.4.46.

```powershell
conda env create -f environment.yml
conda activate ul-transxnet
```

If your CUDA stack differs, install a matching PyTorch build first, then install the remaining dependencies.

## 2. Artifact sanity checks

These checks do not require medical datasets.

```powershell
python scripts/smoke_test.py --model ul-transxnet --num-classes 2 --input-size 256
python scripts/reproduce_main_tables.py --results-dir results --out-dir reproduced_tables
python scripts/validate_artifact.py
```

Expected outcome:

- smoke test returns output shape `[1, 2]`;
- table reproduction writes Markdown versions of the compact CSV tables;
- artifact validation reports no committed dataset, checkpoint, ONNX, APK, or oversized binary files.

## 3. Dataset preparation

Download TN5000, BUSI, and AUL from their original sources. Then adapt the local paths in `configs/*.yaml` or pass command-line arguments directly to the dataset construction scripts:

- `src/scripts/busi_build_voc_roi_dataset_v3.py`
- `src/scripts/aul_build_voc_roi_dataset_v1.py`
- `src/scripts/build_tn5000_yolo_detection_dataset.py`
- `src/scripts/build_voc_yolo_detection_dataset.py`

The generated ROI folders and detector datasets should stay outside git.

## 4. Main ROI classification experiments

The three-dataset UL-TransXNet protocol is summarized in:

- `configs/tn5000_roi_ul_transxnet.yaml`
- `configs/busi_roi_ul_transxnet.yaml`
- `configs/aul_roi_ul_transxnet.yaml`

Representative training entrypoints are:

- `src/scripts/fl_tn5000_roi_compare_multimodel.py`
- `src/scripts/fl_busi_roi_compare_5fold.py`
- `src/scripts/fl_aul_roi_binary_compare_5fold.py`
- `src/scripts/run_withmca_full_three_datasets_detached.py`

## 5. Ablation and automatic ROI experiments

TN5000 current-protocol structure and module ablations are summarized in `configs/tn5000_ablation_current_protocol.yaml` and `results/tn5000_current_protocol_ablation_summary.csv`.

Automatic ROI detector and closed-loop experiments are summarized in `configs/auto_roi_pipeline.yaml` and the workflow scripts under `src/scripts/`.

## 6. Paper tables and figures

The manuscript source is under `paper/`. Compact result tables are under `results/`. Use `scripts/reproduce_main_tables.py` to regenerate Markdown versions of the CSV tables for audit.
