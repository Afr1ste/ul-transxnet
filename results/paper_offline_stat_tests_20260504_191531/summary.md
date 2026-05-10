# Offline paired statistical tests

This directory preserves historical paired-test artifacts. The current
manuscript baseline comparison is recomputed with the frozen paper-log
analysis-label snapshot. Historical embedded labels in prediction CSVs are used
only for mismatch auditing.

Authoritative recomputed source:

```text
results/provenance_release_20260510/predictions/recomputed_paperlog_labels/run_level_recomputed_metrics.csv
```

Current repeated-run stability rows compare UL-TransXNet minus the strongest
non-Ours baseline in the main benchmark table:

| Dataset | Baseline | Metric | n | Delta | Run-delta range | p |
|---|---|---|---:|---:|---|---:|
| TN5000 | ConvNeXt-Tiny | AUC | 3 | +0.0024 | [-0.0024, +0.0051] | 1.0000 |
| TN5000 | ConvNeXt-Tiny | BalAcc | 3 | +0.0058 | [-0.0023, +0.0215] | 1.0000 |
| BUSI | Swin-T | AUC | 5 | +0.0294 | [+0.0179, +0.0422] | 0.0625 |
| BUSI | Swin-T | BalAcc | 5 | +0.0714 | [+0.0460, +0.0989] | 0.0625 |
| AUL | Swin-T | AUC | 5 | -0.0071 | [-0.0189, +0.0203] | 0.3750 |
| AUL | Swin-T | BalAcc | 5 | -0.0376 | [-0.1117, +0.0869] | 0.3750 |
