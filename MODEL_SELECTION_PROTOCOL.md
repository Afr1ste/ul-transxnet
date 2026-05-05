# Model Selection Protocol

This document records the model-family selection protocol used for the current manuscript revision. It is intended to make the final UL-TransXNet choice auditable without relying on current intermediate dataset folders, which may have drifted relative to the frozen result logs.

## Rule

The final TransXNet-family architecture is selected by the highest mean validation AUC over TN5000, BUSI, and AUL under the dataset-specific training protocols:

- TN5000: mean over three seeds.
- BUSI: mean over five training folds evaluated with a fixed held-out test set.
- AUL: mean over five training folds evaluated with a fixed held-out test set.

Test-set metrics are reported after the architecture row is fixed by the validation audit. Validation thresholds, temperatures, and operating points are also derived from validation predictions.

## Audit Result

| Variant | TN5000 val AUC | BUSI val AUC | AUL val AUC | Mean val AUC | Selected |
|---|---:|---:|---:|---:|---|
| TransXNet | 0.9577 | 0.7796 | 0.7464 | 0.8279 | no |
| TransXNet-GG | 0.9632 | 0.8011 | 0.7390 | 0.8345 | no |
| GGG-noMCA | 0.9700 | 0.8957 | 0.8798 | 0.9152 | no |
| UL-TransXNet | 0.9638 | 0.9199 | 0.8827 | 0.9222 | yes |

The machine-readable audit table is `results/no_retrain_revision_20260505/validation_selection_audit.csv`.

## Source Files and Checksums

The source-file index is `results/no_retrain_revision_20260505/model_selection_source_files.csv`. It records, for each candidate/dataset pair, the frozen CSV path, row filter, byte size, last-write time, and SHA256 checksum.

## No-Retrain Boundary

The 2026-05-05 no-retrain revision pass used only frozen completed-result CSVs and existing automatic-ROI summaries. It did not reload current intermediate datasets, split manifests, checkpoints, model weights, or image folders because working dataset folders may have label drift relative to the frozen logs.
