# TN5000 Android ORT Prototype

This Android project documents the on-device evaluation app used for the
teacher/student mobile-feasibility experiments. It is source-only: ONNX files,
APKs, and build outputs are intentionally excluded from the repository.

## Deployment modes

The current source supports exported artifacts for:

- `GGG-MCA`: TransXNet-family teacher artifact.
- `EffFormer-L1+KD`: distilled EfficientFormer-L1 student.
- `EffFormer-L1+ECA+KD`: distilled EfficientFormer-L1 student with ECA.
- Additional baseline ONNX assets can be dropped into `app/src/main/assets/`
  for local latency comparisons.

## Expected model assets

Put ONNX files under `app/src/main/assets/`. The repository only includes
`PLACE_ONNX_MODELS_HERE.txt`; model binaries are omitted because they are large
trained artifacts.

## Headless batch runner

`HeadlessBatchActivity` enables command-line Android evaluation with `adb`.
The CMPB provenance refresh used:

```text
modes=EffFormer-L1+KD,EffFormer-L1+ECA+KD,GGG-MCA
cold_runs=0
hot_runs=5
expand_ratio=0.30
square_crop=false
```

The app can load the frozen paper-log analysis-label snapshot from either:

```text
/data/user/0/com.afriste.tn5000ortdemo/files/paper_log_case_labels.csv
/sdcard/Android/data/com.afriste.tn5000ortdemo/files/paper_log_case_labels.csv
```

The checked-in two-device output summaries are in:

```text
../results/provenance_release_20260510/android_two_device/
```

## Dataset folder layout on the phone

The selected TN5000 root must contain:

```text
Annotations/
JPEGImages/
ImageSets/Main/test.txt
```

The app parses XML boxes, applies the same 30% expanded ROI crop protocol,
runs ONNX Runtime inference, and exports per-run CSV/TXT summaries under the
app external files directory.
