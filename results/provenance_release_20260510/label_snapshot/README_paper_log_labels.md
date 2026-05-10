# Paper log label reconstruction

Generated at: 2026-05-05T21:23:59

This snapshot reconstructs the label set used by the log-matched paper experiments. It is non-destructive: no live dataset label files were overwritten.

## Source experiment logs

- TN5000: `tn5000_compare_5models_3seed_logs/20260402_192605`
- BUSI: `busi_compare_5models_5fold_logs/20260403_083238`
- AUL: `aul_compare_extra4models_5fold_logs/20260421_085028`

Note: `tn5000_ggg_mca_enabled_3seed_logs/20260426_093728` was checked but not used for this snapshot because its log counts are 1416/3584, not the expected 1435/3565.

TN5000 train labels are the only partial exception: the paper log did not export train predictions, so train labels were read from the current TN5000 train XML only after its train count matched the log exactly.

## Summary

| Dataset | Split | Unique cases | Benign | Malignant | Expected all-count match |
|---|---|---:|---:|---:|---|
| AUL | all | 635 | 183 | 452 | True |
| AUL | trainval | 508 | 139 | 369 |  |
| AUL | test | 127 | 44 | 83 |  |
| BUSI | all | 647 | 435 | 212 | True |
| BUSI | trainval | 518 | 344 | 174 |  |
| BUSI | test | 129 | 91 | 38 |  |
| TN5000 | all | 5000 | 1435 | 3565 | True |
| TN5000 | train | 3500 | 1032 | 2468 |  |
| TN5000 | val | 500 | 122 | 378 |  |
| TN5000 | test | 1000 | 281 | 719 |  |

- Label conflicts found: 0

## Files

- `paper_log_label_observations.csv`: all source observations.
- `paper_log_case_labels.csv`: unique reconstructed labels.
- `paper_log_label_conflicts.csv`: any label conflicts across sources.
- `paper_log_label_sources.csv`: source files and SHA256 hashes.
- `paper_log_label_summary.csv`: class counts by dataset and split view.
- `paper_log_count_lines_from_logs.csv`: parsed split-level label count lines from the selected log directories.
- `labels_by_dataset/*/*_paper_log_labels.csv`: compact per-dataset label files.