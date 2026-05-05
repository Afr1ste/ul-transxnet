# High-ROI no-retrain revision outputs

Generated at: 2026-05-05T20:53:13

All outputs are generated from frozen prediction/result CSV files only. No current dataset manifests, labels, images, checkpoints, or training scripts are loaded.

## TN5000 oracle/auto/full probe

| Input | AUC | BalAcc | Sens. | Spec. | F1 | Acc |
|---|---:|---:|---:|---:|---:|---:|
| Oracle ROI | 0.9464 | 0.8725 | 0.8387 | 0.9062 | 0.8299 | 0.8560 |
| Automatic ROI | 0.9391 | 0.8675 | 0.8522 | 0.8828 | 0.8320 | 0.8600 |
| Full image | 0.6490 | 0.5997 | 0.7110 | 0.4883 | 0.5865 | 0.6540 |

## TN5000 localization-robustness probes

| Probe | Input | AUC | BalAcc | F1 | Acc |
|---|---|---:|---:|---:|---:|
| Box-jitter probe | Oracle ROI | 0.9510 | 0.8748 | 0.8236 | 0.8480 |
| Box-jitter probe | Automatic ROI | 0.9457 | 0.8739 | 0.8354 | 0.8620 |
| Box-jitter probe | Full image | 0.6387 | 0.5923 | 0.5605 | 0.6030 |
| Predicted-box mix | Oracle ROI | 0.9466 | 0.8840 | 0.8511 | 0.8770 |
| Predicted-box mix | Automatic ROI | 0.9429 | 0.8740 | 0.8372 | 0.8640 |
| Predicted-box mix | Full image | 0.6598 | 0.6097 | 0.5977 | 0.6670 |

## Case-level diagnostic statistics

| Dataset | n | AUC | Sens. | Spec. | PPV | NPV | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TN5000 | 1000 | 0.9540 | 0.8750 | 0.8828 | 0.9559 | 0.7085 | 0.0237 | 0.0683 |
| BUSI | 129 | 0.9023 | 0.7115 | 0.8961 | 0.8222 | 0.8214 | 0.1035 | 0.1317 |
| AUL | 127 | 0.8752 | 0.7660 | 1.0000 | 1.0000 | 0.6000 | 0.1041 | 0.1558 |

## Generated files

- `tn5000_oracle_auto_full_probe.csv`
- `tn5000_oracle_auto_full_probe_table.tex`
- `tn5000_localization_robustness_probe.csv`
- `tn5000_localization_robustness_probe_table.tex`
- `case_level_diagnostic_statistics.csv`
- `case_level_diagnostic_statistics_table.tex`
- `reliability_curve_bins_case_level.csv`
- `fig_reliability_diagram_case_level.png`