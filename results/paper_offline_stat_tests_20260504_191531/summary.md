# Offline paired statistical tests

Comparison: UL-TransXNet minus strongest non-Ours baseline from the main result table.

| dataset   | baseline      | metric   |   n_pairs |   ours_mean |   baseline_mean |   delta_mean |      ci_low |      ci_high |   p_signflip |
|:----------|:--------------|:---------|----------:|------------:|----------------:|-------------:|------------:|-------------:|-------------:|
| TN5000    | ConvNeXt-Tiny | AUC      |         3 |    0.948295 |        0.95084  |  -0.00254483 | -0.00404417 | -0.000558846 |       0.25   |
| TN5000    | ConvNeXt-Tiny | BalAcc   |         3 |    0.872564 |        0.87939  |  -0.0068266  | -0.0103667  | -0.00216702  |       0.25   |
| BUSI      | Swin-T        | AUC      |         5 |    0.89975  |        0.819028 |   0.0807219  |  0.069654   |  0.0917898   |       0.0625 |
| BUSI      | Swin-T        | BalAcc   |         5 |    0.796079 |        0.713736 |   0.0823427  |  0.0646827  |  0.0960802   |       0.0625 |
| AUL       | Swin-T        | AUC      |         5 |    0.876725 |        0.831873 |   0.0448517  |  0.0374648  |  0.0574254   |       0.0625 |
| AUL       | Swin-T        | BalAcc   |         5 |    0.820019 |        0.778313 |   0.0417061  |  0.0159809  |  0.0674312   |       0.0625 |

## Sources
- TN5000 ours: `<LOCAL_THYROID_ROOT>\tn5000_ggg_mca_enabled_3seed_logs\20260426_093728\all_runs_metrics.csv`
- TN5000 baseline: `<LOCAL_THYROID_ROOT>\tn5000_compare_extra4models_3seed_logs\20260421_222342_merged_complete\all_runs_metrics.csv`
- BUSI ours: `<LOCAL_THYROID_ROOT>\busi_ggg_mca_clean_5fold_safe_logs\20260426_165332\all_runs_metrics.csv`
- BUSI baseline: `<LOCAL_THYROID_ROOT>\busi_compare_5models_5fold_logs\20260403_083238\all_runs_metrics.csv`
- AUL ours: `<LOCAL_THYROID_ROOT>\aul_ggg_mca_clean_5fold_safe_logs\20260426_200618\all_runs_metrics.csv`
- AUL baseline: `<LOCAL_THYROID_ROOT>\aul_roi_compare_5models_5fold_logs\20260404_235703\all_runs_metrics.csv`
