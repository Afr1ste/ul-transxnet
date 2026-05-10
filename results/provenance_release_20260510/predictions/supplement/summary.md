# Paper-log P1/P2 Supplement

Generated from completed prediction CSVs only. The label source is the paper-log snapshot; manuscript Table 2 counts are intentionally not modified.

## P1 Case-bootstrap Summary

| Dataset | Variant | n | AUC | BalAcc | F1 | ECE |
|---|---|---:|---:|---:|---:|---:|
| TN5000 | BASE_tn5000_p1_current | 1000 | 0.9470 [0.9302, 0.9616] | 0.8738 [0.8519, 0.8935] | 0.8369 | 0.0300 |
| TN5000 | DA_tn5000_p1_current | 1000 | 0.9467 [0.9319, 0.9611] | 0.8801 [0.8570, 0.9040] | 0.8810 | 0.0229 |
| TN5000 | MUDD_DA_MCA_tn5000_p1_current | 1000 | 0.9566 [0.9426, 0.9694] | 0.8927 [0.8698, 0.9139] | 0.8745 | 0.0200 |
| TN5000 | MUDD_DA_NOMCA_tn5000_p1_current | 1000 | 0.9618 [0.9484, 0.9733] | 0.9038 [0.8828, 0.9241] | 0.8918 | 0.0179 |
| TN5000 | MUDD_tn5000_p1_current | 1000 | 0.9576 [0.9438, 0.9700] | 0.8909 [0.8684, 0.9127] | 0.8863 | 0.0173 |
| BUSI | BASE_busi_p1_clean | 129 | 0.8259 [0.7294, 0.9092] | 0.7695 [0.6783, 0.8508] | 0.7529 | 0.0622 |
| BUSI | DA_busi_p1_clean | 129 | 0.8404 [0.7477, 0.9227] | 0.8036 [0.7248, 0.8814] | 0.8154 | 0.1005 |
| BUSI | MUDD_DA_busi_p1_clean | 129 | 0.8239 [0.7282, 0.9075] | 0.7629 [0.6797, 0.8397] | 0.7297 | 0.1329 |
| BUSI | MUDD_busi_p1_clean | 129 | 0.8285 [0.7311, 0.9170] | 0.7882 [0.7081, 0.8620] | 0.7705 | 0.0811 |
| AUL | BASE_aul_p1_clean | 127 | 0.7897 [0.6897, 0.8816] | 0.7927 [0.7155, 0.8694] | 0.8019 | 0.0792 |
| AUL | DA_aul_p1_clean | 127 | 0.8004 [0.7002, 0.8859] | 0.7177 [0.6339, 0.7940] | 0.6924 | 0.1054 |
| AUL | MUDD_DA_aul_p1_clean | 127 | 0.8311 [0.7399, 0.9080] | 0.8321 [0.7582, 0.9012] | 0.8339 | 0.1716 |
| AUL | MUDD_aul_p1_clean | 127 | 0.8209 [0.7295, 0.9025] | 0.8101 [0.7366, 0.8819] | 0.8199 | 0.1209 |

## P2 Oracle-threshold Transfer

This checks whether the automatic/full-image input modes still work when they reuse the oracle-ROI validation threshold instead of selecting an input-mode-specific threshold.

| Variant | Mode | Split | Policy | AUC | BalAcc | F1 | recall_0 | recall_1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| mudd_da_mca | auto | test | native_mode_threshold | 0.9484 | 0.8796 | 0.8658 | 0.8612 | 0.8980 |
| mudd_da_mca | auto | test | oracle_threshold_transfer | 0.9484 | 0.8788 | 0.8568 | 0.8814 | 0.8762 |
| mudd_da_mca | full | test | native_mode_threshold | 0.6619 | 0.6077 | 0.5295 | 0.7758 | 0.4395 |
| mudd_da_mca | full | test | oracle_threshold_transfer | 0.6619 | 0.5893 | 0.4515 | 0.9004 | 0.2782 |
| mudd_da_nomca | auto | test | native_mode_threshold | 0.9500 | 0.8903 | 0.8653 | 0.9039 | 0.8767 |
| mudd_da_nomca | auto | test | oracle_threshold_transfer | 0.9500 | 0.8895 | 0.8841 | 0.8541 | 0.9249 |
| mudd_da_nomca | full | test | native_mode_threshold | 0.6869 | 0.6245 | 0.6103 | 0.5480 | 0.7010 |
| mudd_da_nomca | full | test | oracle_threshold_transfer | 0.6869 | 0.5949 | 0.4718 | 0.8754 | 0.3143 |
