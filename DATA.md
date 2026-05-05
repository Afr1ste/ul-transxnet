# Data access and local layout

This repository does not redistribute medical images or annotations. The datasets used in the manuscript must be obtained from their original public sources and cited according to their original licenses and terms.

## Datasets used

- TN5000: thyroid nodule ultrasound classification experiments and TN5000 automatic ROI experiments.
- BUSI: breast ultrasound ROI classification and one-class lesion detector experiments.
- AUL: abdominal ultrasound liver lesion binary ROI classification and one-class lesion detector experiments.

## Expected local layout

The original experiments used dataset roots with the following logical content:

```text
<data_root>/
  images or image folders
  annotation files, masks, or VOC-style XML files
  train/validation/test split files or generated fold files
```

Portable config templates are provided under `configs/`. Set local paths there or pass equivalent command-line arguments to the scripts under `src/scripts/`.

## Important protocol note

The main classification protocol uses annotation-derived ROI crops. The automatic ROI experiments are separate closed-loop experiments that train a detector to predict lesion boxes and then feed expanded square crops to the classifier. The repository keeps these protocols separate to avoid conflating manually defined ROI classification with full-image automatic inference.

## Non-redistribution policy

Do not commit:

- raw medical images;
- dataset archives;
- generated ROI image folders;
- trained `.pth` or `.pt` checkpoints;
- ONNX model binaries;
- Android APKs;
- patient-level metadata beyond what the original dataset permits.
