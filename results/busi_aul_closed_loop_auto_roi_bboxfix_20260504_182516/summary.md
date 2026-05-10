# BUSI/AUL closed-loop ROI classification summary

## BUSI
### oracle
- auc: 0.8998 +/- 0.0129
- bal_acc: 0.7961 +/- 0.0116
- f1_macro: 0.8018 +/- 0.0133
- acc: 0.8140 +/- 0.0145
- recall_0: 0.8883 +/- 0.0407
- recall_1: 0.7038 +/- 0.0375

### auto
- auc: 0.8725 +/- 0.0177
- bal_acc: 0.7839 +/- 0.0201
- f1_macro: 0.7893 +/- 0.0197
- acc: 0.8031 +/- 0.0187
- recall_0: 0.8831 +/- 0.0511
- recall_1: 0.6846 +/- 0.0661

### full
- auc: 0.7737 +/- 0.0049
- bal_acc: 0.6982 +/- 0.0265
- f1_macro: 0.6915 +/- 0.0422
- acc: 0.7023 +/- 0.0467
- recall_0: 0.7195 +/- 0.1413
- recall_1: 0.6769 +/- 0.1012

## AUL
### oracle
- auc: 0.8767 +/- 0.0011
- bal_acc: 0.8200 +/- 0.0545
- f1_macro: 0.7462 +/- 0.0453
- acc: 0.7685 +/- 0.0468
- recall_0: 0.9273 +/- 0.1463
- recall_1: 0.7128 +/- 0.0851

### auto
- auc: 0.8507 +/- 0.0248
- bal_acc: 0.7593 +/- 0.0323
- f1_macro: 0.6908 +/- 0.0627
- acc: 0.7165 +/- 0.0797
- recall_0: 0.8485 +/- 0.1469
- recall_1: 0.6702 +/- 0.1523

### full
- auc: 0.7441 +/- 0.0124
- bal_acc: 0.6805 +/- 0.0239
- f1_macro: 0.6787 +/- 0.0227
- acc: 0.7512 +/- 0.0213
- recall_0: 0.5333 +/- 0.0550
- recall_1: 0.8277 +/- 0.0331

## Outputs
- per-fold CSV: `<LOCAL_THYROID_ROOT>\eval_reports\busi_aul_closed_loop_auto_roi_bboxfix_20260504_182516\closed_loop_per_fold.csv`
- aggregate CSV: `<LOCAL_THYROID_ROOT>\eval_reports\busi_aul_closed_loop_auto_roi_bboxfix_20260504_182516\closed_loop_aggregate.csv`