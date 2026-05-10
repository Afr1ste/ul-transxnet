# BUSI/AUL closed-loop ROI classification summary

This directory preserves the historical closed-loop prediction CSVs. The
current manuscript does not use the embedded labels in those CSVs directly.
Metrics are recomputed with the frozen paper-log analysis-label snapshot by:

```text
scripts/recompute_paperlog_label_metrics.py
```

Authoritative recomputed outputs:

```text
results/provenance_release_20260510/predictions/recomputed_paperlog_labels/auto_roi_recomputed.csv
results/provenance_release_20260510/predictions/recomputed_paperlog_labels/auto_roi_label_mismatch_audit.csv
```

Current manuscript rows:

| Dataset | Input | AUC | BalAcc | F1 | Acc |
|---|---|---:|---:|---:|---:|
| BUSI | Oracle ROI | 0.8485 | 0.7851 | 0.7695 | 0.7984 |
| BUSI | Detector ROI | 0.8393 | 0.7809 | 0.7667 | 0.7969 |
| BUSI | Full image | 0.7575 | 0.7064 | 0.6710 | 0.6961 |
| AUL | Oracle ROI | 0.8248 | 0.7407 | 0.7215 | 0.7323 |
| AUL | Detector ROI | 0.8094 | 0.7358 | 0.7124 | 0.7244 |
| AUL | Full image | 0.7104 | 0.6268 | 0.6307 | 0.6866 |
