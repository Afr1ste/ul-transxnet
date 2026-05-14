# Artifact and Manifest Requirements Before Submission

This note records the reproducibility artifacts needed to fully defend the current manuscript during review or revision. The lightweight public repository contains compact table-level summaries; the full internal audit package should retain the case-level files listed below.

## Why this matters

The manuscript reports fixed split/protocol metadata and final aggregate tables for robustness, mobile, and cross-protocol analyses. Aggregate counts alone are not sufficient for a revision or data audit. The corresponding author should be able to trace each table back to a file-level manifest and case-level predictions, even when those files are not redistributed in the lightweight public repository.

## Required manifest columns

Each dataset-level manifest should include:

- `case_id`: stable unique identifier used in prediction CSV files.
- `dataset`: TN5000, BUSI, or AUL.
- `relative_image_path`: path relative to the dataset root, without redistributing the image itself.
- `original_label`: label from the source dataset or source annotation.
- `label`: label used by the reported fixed protocol.
- `split`: train, val, test, trainval, or fold-specific split name.
- `fold`: fold index or empty for non-folded protocols.
- `bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax`: lesion box if available.
- `roi_expand_ratio`: crop expansion ratio used by the corresponding protocol.
- `excluded`: true or false.
- `exclusion_reason`: empty unless a case is excluded.
- `source_manifest_hash`: hash of the upstream/source manifest or annotation file.
- `manifest_hash`: hash of the fixed split/protocol manifest.

## Required prediction CSV columns

Each table-generating prediction CSV should include:

- `case_id`
- `dataset`
- `split`
- `fold_or_seed`
- `model_name`
- `protocol`
- `y_true`
- `prob_benign`
- `prob_malignant`
- `pred_label`
- `threshold`
- `temperature`
- `checkpoint_id` or `ensemble_member`
- `manifest_hash`

## Required table provenance

For every reported table, keep a small provenance record with:

- table identifier, e.g. `Table 2`, `Table 5`, `Table 6`.
- manifest hash.
- prediction CSV path.
- metric script path.
- timestamp.
- git commit or source archive hash if available.
- exact label-count summary produced from the manifest.

## Submission gate

Before or during review, the corresponding author should be able to regenerate each numerical table from:

1. the file-level manifest,
2. the prediction CSV files,
3. the metric aggregation script,
4. the table-generation script.

If this chain is missing, the manuscript should still be treated as scientifically incomplete even if the PDF is polished.
