# Reproducibility Statement

The checked-in package is a CSV-level reproducibility and provenance release.
It is designed to let readers audit manuscript tables, inspect the frozen
analysis-label snapshot, recompute case-level metrics from prediction CSVs,
validate model-selection source checksums, inspect scripts, and rebuild the
LaTeX manuscript and supplementary material.

Full retraining requires independently obtained TN5000, BUSI, and AUL source
data under their original licenses. The 2026-05-05 revision intentionally did
not reload current intermediate dataset folders because local labels and
generated derivatives may have drifted relative to the frozen completed-run
outputs used in the manuscript. The 2026-05-10 provenance release adds a
sanitized public case manifest, per-case averaged predictions, ROI-robustness
prediction CSVs, BUSI duplicate-audit outputs, and two-device Android summaries
without redistributing image pixels or trained binaries.
