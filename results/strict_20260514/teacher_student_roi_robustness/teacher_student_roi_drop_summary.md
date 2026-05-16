# Teacher-student ROI robustness summary

This inference-only report evaluates the final high-capacity teachers and mobile students under the same TN5000 ROI perturbation protocol.

| Model | GT AUC | 20% AUC | Drop@20% | Detector AUC | Full-image AUC |
|---|---:|---:|---:|---:|---:|
| EfficientFormer-L1+ECA student | 0.9280 | 0.9258 | +0.0021 | 0.9308 | 0.7165 |
| EfficientFormer-L1+ECA+KD student | 0.9557 | 0.9513 | +0.0043 | 0.9523 | 0.7111 |
| EfficientFormer-L1+KD student | 0.9585 | 0.9529 | +0.0056 | 0.9534 | 0.7063 |
| TransXNet-MUDD+DA teacher | 0.9623 | 0.9463 | +0.0160 | 0.9497 | 0.6869 |
| UL-TransXNet teacher | 0.9600 | 0.9376 | +0.0224 | 0.9487 | 0.6556 |
