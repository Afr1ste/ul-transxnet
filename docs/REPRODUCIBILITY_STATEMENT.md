# Reproducibility Statement

The checked-in package is a CSV-level reproducibility and provenance release.
It is designed to let readers audit manuscript tables, validate model-selection
source checksums, inspect scripts, and rebuild the LaTeX manuscript.

Full retraining requires independently obtained TN5000, BUSI, and AUL source
data under their original licenses. The 2026-05-05 revision intentionally did
not reload current intermediate dataset folders because local labels and
generated derivatives may have drifted relative to the frozen completed-run
outputs used in the manuscript.
