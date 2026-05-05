# TN5000 Android ORT Demo Complete

This project is a full on-device TN5000 evaluation app with three deployment tiers:

1. **Fast**: single ONNX, no TTA, temperature scaling + threshold.
2. **Accurate**: single ONNX + horizontal-flip TTA + temperature scaling + threshold.
3. **Ensemble**: three ONNX checkpoints + horizontal-flip TTA + temperature scaling + threshold.

## Expected model assets

Put these files under `app/src/main/assets/`:

- `tn5000_current_mainline.onnx`  ← required for Fast / Accurate
- `tn5000_epoch060_bal_acc_0.9195.onnx` ← optional, required for full Ensemble
- `tn5000_epoch054_bal_acc_0.9169.onnx` ← optional, required for full Ensemble

`tn5000_current_mainline.onnx` is assumed to correspond to your main exported checkpoint.

## Training-aligned defaults

Use these defaults to match the current desktop mainline as closely as possible:

- expand ratio = `0.30`
- square crop = `false`
- threshold = `0.61`
- temperature = `1.157835`

## Dataset folder layout on the phone

Select the root folder that contains:

- `Annotations/`
- `JPEGImages/`
- `ImageSets/Main/test.txt`

The app parses XML labels + bboxes, crops ROI on-device, runs the selected deployment mode, exports CSV/TXT reports, and shows paper-friendly statistics including accuracy / balanced accuracy / macro-F1 / AUC / latency breakdown / memory estimates.


Paper-batch collection
----------------------
This build adds a paper-oriented batch runner:
- select one or more deployment modes (Fast / Accurate / Ensemble)
- set cold runs per mode and hot runs per mode
- run the entire TN5000 test split repeatedly on-device
- export per-run CSV/TXT plus aggregate CSV summaries under `Android/data/com.afriste.tn5000ortdemo/files/eval_reports/paper_batch_*`
