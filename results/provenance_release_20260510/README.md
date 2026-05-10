# Provenance release 2026-05-10

This directory contains the public, CSV-level provenance package for the CMPB
submission draft. It includes the frozen paper-log label snapshot, public case
manifest, per-case averaged predictions, recomputed paper-log-label benchmark
tables, ROI robustness prediction CSVs, BUSI duplicate-audit tables,
two-device Android summaries, and SHA256 hashes.

No dataset images, generated ROI image folders, checkpoints, detector weights,
ONNX binaries, APKs, or raw training logs are included.

Important boundaries:

- `label_snapshot/paper_log_case_manifest_public.csv` is the authoritative
  label/split snapshot for the added analyses.
- `predictions/paperlog_per_case_averaged_predictions_public.csv` is the main
  per-case averaged prediction table used for case-level statistics.
- `predictions/recomputed_paperlog_labels/` contains the current manuscript
  benchmark tables recomputed by replacing historical prediction-file labels
  with the frozen paper-log label snapshot. Its label-mismatch audit records
  where old logs carried stale embedded labels.
- `predictions/roi_robustness/` contains TN5000 test prediction CSVs for the
  GT/noisy/detector ROI robustness benchmark.
- `android_two_device/` contains Xiaomi and Samsung hot-run summaries.
- `release_manifest.csv` records size and SHA256 for every file in this
  release directory.
