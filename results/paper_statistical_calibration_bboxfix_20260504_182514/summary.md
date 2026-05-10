# Statistical and calibration supplement

This directory preserves historical statistical-calibration artifacts. The
current manuscript values are recomputed with the frozen paper-log
analysis-label snapshot instead of trusting embedded labels in older prediction
CSVs.

Authoritative recomputed outputs:

```text
results/provenance_release_20260510/predictions/recomputed_paperlog_labels/calibration_uncertainty_recomputed.csv
results/provenance_release_20260510/predictions/recomputed_paperlog_labels/case_level_ul_diagnostics_recomputed.csv
```

Current case-level calibration rows:

| Dataset | AUC 95% CI | BalAcc 95% CI | ECE | Brier |
|---|---|---|---:|---:|
| TN5000 | 0.960 [0.946, 0.973] | 0.909 [0.892, 0.930] | 0.0226 | 0.0662 |
| BUSI | 0.850 [0.756, 0.932] | 0.800 [0.746, 0.884] | 0.0538 | 0.1770 |
| AUL | 0.826 [0.733, 0.903] | 0.832 [0.760, 0.902] | 0.1474 | 0.1979 |
