# Reproduction Notes

Typical workflow:

1. Download TN5000, BUSI, and AUL from their official/public sources.
2. Build ROI datasets with the dataset-construction scripts in `src/scripts/`.
3. Train UL-TransXNet and baselines with the corresponding run scripts.
4. Evaluate automatic ROI detection and closed-loop classification with the detector evaluation scripts.
5. Regenerate paper statistics and figures with `generate_paper_*` and `src/tools/*`.

The original experiments were run on a local Windows workstation with CUDA. Some scripts contain absolute local paths from the experiment environment; adjust them before running elsewhere.
