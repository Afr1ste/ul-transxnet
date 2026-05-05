# Reproducibility statement

This repository follows a lightweight reproducibility artifact structure suitable for manuscript review:

1. The code that defines the proposed model and baselines is included.
2. The data acquisition boundary is explicit: raw medical datasets are not redistributed.
3. Dataset construction scripts and protocol configs are included so derived ROI folders can be rebuilt after obtaining the data.
4. Compact result CSV files used by the manuscript are included.
5. A table reproduction script regenerates readable audit tables from the compact CSV files.
6. A smoke test validates model construction and forward inference without requiring private or restricted data.
7. A hygiene validator checks that restricted or large artifacts were not accidentally committed.

The repository is therefore intended for protocol audit, source-code inspection, table traceability, and local reproduction by readers who have obtained the datasets. It is not intended to be a one-command benchmark without dataset setup, because the original medical image datasets have independent access and redistribution conditions.
