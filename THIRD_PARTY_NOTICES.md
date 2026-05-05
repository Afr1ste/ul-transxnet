# Third-party notices and provenance

This repository combines newly written experiment code with model implementations and ideas from public research projects. The manuscript bibliography gives the formal scholarly citations. This file is a practical provenance map for readers auditing the codebase.

## Neural network backbones and libraries

- TransXNet-family implementation: used as the base family for the proposed UL-TransXNet variants. The published model files include local adaptations for the ultrasound ROI experiments and ablation variants.
- EdgeNeXt implementation: included for baseline and comparison support.
- RepViT implementation: included for baseline and comparison support.
- timm: used for model registration, pretrained backbones, and selected layers.
- PyTorch / torchvision: used as the core deep learning framework.
- MMCV / MMDetection / MMSegmentation-compatible imports: retained where required by the TransXNet-family implementation.
- Ultralytics YOLO: used for the auxiliary automatic ROI detector experiments.

## Datasets

The repository does not include TN5000, BUSI, AUL, or any derived medical images. Dataset use must follow the original dataset terms and citation requirements.

## Redistribution boundary

The public repository includes code, configs, compact CSV summaries, paper source, and selected final figures. It excludes trained checkpoints, detector weights, ONNX files, APK files, raw datasets, generated ROI folders, and full training logs.

## Reviewer note

Some historical scripts under `src/scripts/` preserve the local paths used in the original experiments. Portable path templates are provided in `configs/`, and the public artifact checks treat such paths as historical provenance rather than required public paths.
