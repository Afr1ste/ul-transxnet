# Statistical and calibration supplement

Output directory: `C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_reports\paper_statistical_calibration_bboxfix_20260504_182514`

## Main UL-TransXNet uncertainty/calibration
- TN5000: AUC 0.9483 [0.9475, 0.9488], BalAcc 0.8726 [0.8662, 0.8771], ECE 0.0220, Brier 0.0741
- BUSI: AUC 0.8998 [0.8880, 0.9067], BalAcc 0.7961 [0.7865, 0.8045], ECE 0.0696, Brier 0.1358
- AUL: AUC 0.8767 [0.8759, 0.8776], BalAcc 0.8200 [0.7755, 0.8599], ECE 0.1283, Brier 0.1597

## Automatic ROI key paired deltas
- BUSI auto-full: delta AUC 0.0988 [0.0833, 0.1100], delta BalAcc 0.0857 [0.0592, 0.1099]
- BUSI auto-oracle: delta AUC -0.0273 [-0.0321, -0.0224], delta BalAcc -0.0122 [-0.0323, 0.0121]
- AUL auto-full: delta AUC 0.1066 [0.0866, 0.1287], delta BalAcc 0.0789 [0.0621, 0.0981]
- AUL auto-oracle: delta AUC -0.0260 [-0.0447, -0.0066], delta BalAcc -0.0607 [-0.0842, -0.0337]
