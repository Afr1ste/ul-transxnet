# Main Result Statistics

- Bootstraps: 3000
- Output dir: C:\Users\Afr1ste\PycharmProjects\Thyroid\eval_reports\main_result_statistics_20260421_045323

## Dataset Sources
- TN5000: C:\Users\Afr1ste\PycharmProjects\Thyroid\tn5000_compare_5models_3seed_logs\20260402_192605
- BUSI: C:\Users\Afr1ste\PycharmProjects\Thyroid\busi_compare_5models_5fold_logs\20260403_083238
- AUL: C:\Users\Afr1ste\PycharmProjects\Thyroid\aul_roi_compare_5models_5fold_logs\20260404_235703

## CI Summary
- AUL | Swin-T: AUC 0.8368 [0.7514, 0.9075], BalAcc 0.7793 [0.6984, 0.8549], F1 0.7763 [0.6973, 0.8497]
- AUL | Ours: AUC 0.8215 [0.7314, 0.9039], BalAcc 0.7987 [0.7192, 0.8744], F1 0.8097 [0.7299, 0.8837]
- AUL | MobileNetV3-Large: AUC 0.7388 [0.6427, 0.8223], BalAcc 0.6889 [0.6034, 0.7699], F1 0.6672 [0.5860, 0.7445]
- AUL | EfficientNet-B0: AUC 0.6435 [0.5457, 0.7413], BalAcc 0.5831 [0.5122, 0.6540], F1 0.5008 [0.4202, 0.5748]
- AUL | ResNet50: AUC 0.5553 [0.4422, 0.6684], BalAcc 0.4895 [0.4132, 0.5724], F1 0.4022 [0.3262, 0.4836]
- BUSI | Ours: AUC 0.8528 [0.7677, 0.9257], BalAcc 0.7739 [0.6938, 0.8485], F1 0.7440 [0.6686, 0.8158]
- BUSI | Swin-T: AUC 0.8256 [0.7368, 0.8985], BalAcc 0.7509 [0.6619, 0.8298], F1 0.7352 [0.6509, 0.8103]
- BUSI | MobileNetV3-Large: AUC 0.8126 [0.7293, 0.8890], BalAcc 0.7387 [0.6575, 0.8188], F1 0.7058 [0.6283, 0.7856]
- BUSI | EfficientNet-B0: AUC 0.7820 [0.6912, 0.8681], BalAcc 0.7234 [0.6368, 0.8112], F1 0.6998 [0.6184, 0.7832]
- BUSI | ResNet50: AUC 0.7794 [0.6784, 0.8687], BalAcc 0.6914 [0.6080, 0.7715], F1 0.6387 [0.5599, 0.7153]
- TN5000 | Ours: AUC 0.9630 [0.9497, 0.9750], BalAcc 0.9149 [0.8934, 0.9346], F1 0.9126 [0.8927, 0.9310]
- TN5000 | Swin-T: AUC 0.9514 [0.9358, 0.9655], BalAcc 0.8866 [0.8644, 0.9072], F1 0.8723 [0.8508, 0.8940]
- TN5000 | MobileNetV3-Large: AUC 0.8385 [0.8116, 0.8641], BalAcc 0.7622 [0.7330, 0.7920], F1 0.7500 [0.7227, 0.7777]
- TN5000 | EfficientNet-B0: AUC 0.7810 [0.7502, 0.8117], BalAcc 0.6946 [0.6642, 0.7276], F1 0.6793 [0.6505, 0.7103]
- TN5000 | ResNet50: AUC 0.6579 [0.6174, 0.6983], BalAcc 0.6419 [0.6095, 0.6742], F1 0.6368 [0.6058, 0.6679]

## Ours vs Best Non-Ours
- AUL: baseline=Swin-T, DeltaAUC=-0.0144 (p=0.7053), DeltaBalAcc=0.0205 (p=0.5353), DeltaF1=0.0342 (p=0.3307)
- BUSI: baseline=Swin-T, DeltaAUC=0.0278 (p=0.356), DeltaBalAcc=0.0236 (p=0.5807), DeltaF1=0.0093 (p=0.8307)
- TN5000: baseline=Swin-T, DeltaAUC=0.0116 (p=0.08333), DeltaBalAcc=0.0283 (p=0.02333), DeltaF1=0.0404 (p=0.001333)