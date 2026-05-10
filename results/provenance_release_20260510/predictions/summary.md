# Remaining P1/P2 offline artifacts

These outputs are generated from completed prediction CSVs and existing complexity metadata.

- Per-case averaged predictions: `paperlog_per_case_averaged_predictions.csv`
- Metric summary: `paperlog_per_case_metric_summary.csv`
- BUSI/AUL fixed-specificity operating points: `busi_aul_fixed_specificity_operating_points.csv`
- BUSI/AUL ROC operating curve points: `busi_aul_roc_operating_curve_points.csv`
- Method/module detail table: `method_module_detail_table.csv`

## Main Metric Summary

| Dataset | Variant | AUC | BalAcc | Sens | Spec | PPV | NPV |
|---|---|---:|---:|---:|---:|---:|---:|
| TN5000 | BASE_tn5000_p1_current | 0.9470 | 0.8738 | 0.8331 | 0.9146 | 0.9615 | 0.6817 |
| TN5000 | DA_tn5000_p1_current | 0.9467 | 0.8801 | 0.9346 | 0.8256 | 0.9320 | 0.8315 |
| TN5000 | MUDD_DA_MCA_tn5000_p1_current | 0.9566 | 0.8927 | 0.8957 | 0.8897 | 0.9541 | 0.7692 |
| TN5000 | MUDD_DA_NOMCA_tn5000_p1_current | 0.9618 | 0.9038 | 0.9179 | 0.8897 | 0.9551 | 0.8091 |
| TN5000 | MUDD_tn5000_p1_current | 0.9576 | 0.8909 | 0.9277 | 0.8541 | 0.9421 | 0.8219 |
| BUSI | BASE_busi_p1_clean | 0.8259 | 0.7695 | 0.7368 | 0.8022 | 0.6087 | 0.8795 |
| BUSI | DA_busi_p1_clean | 0.8404 | 0.8036 | 0.6842 | 0.9231 | 0.7879 | 0.8750 |
| BUSI | MUDD_DA_busi_p1_clean | 0.8239 | 0.7629 | 0.7895 | 0.7363 | 0.5556 | 0.8933 |
| BUSI | MUDD_busi_p1_clean | 0.8285 | 0.7882 | 0.7632 | 0.8132 | 0.6304 | 0.8916 |
| AUL | BASE_aul_p1_clean | 0.7897 | 0.7927 | 0.9036 | 0.6818 | 0.8427 | 0.7895 |
| AUL | DA_aul_p1_clean | 0.8004 | 0.7177 | 0.6627 | 0.7727 | 0.8462 | 0.5484 |
| AUL | MUDD_DA_aul_p1_clean | 0.8311 | 0.8321 | 0.8916 | 0.7727 | 0.8810 | 0.7907 |
| AUL | MUDD_aul_p1_clean | 0.8209 | 0.8101 | 0.9157 | 0.7045 | 0.8539 | 0.8158 |
