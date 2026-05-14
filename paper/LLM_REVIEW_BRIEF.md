# LLM Review Brief for CMPB Submission

Please review the manuscript as a strict external reviewer for Computer Methods and Programs in Biomedicine (CMPB), not as a coauthor.

## Primary Files

1. `main.pdf` is the current manuscript to review first.
2. `supplementary_material.pdf` is the current supplementary material.
3. `main.tex`, `supplementary_material.tex`, and `refs.bib` are included for source-level checks.
4. `results/strict_20260514/` contains the strict-label result CSVs used by the current manuscript.

## Intended Journal and Claim Boundary

Target journal: Computer Methods and Programs in Biomedicine.

The paper should be judged as a reproducible computing-methods / applied biomedical engineering paper, not as a claim of a fundamentally new vision backbone. The intended contribution is:

- manifest-locked ROI ultrasound lesion-classification evaluation;
- TransXNet-family teacher design-space analysis;
- localization robustness under GT, perturbed, detector-derived, and full-image inputs;
- EfficientFormer-L1 mobile distillation and two-device Android measurement;
- explicit separation between multi-organ in-domain benchmarking and weak zero-shot leave-one-domain-out transfer.

The paper should not claim clinical deployment readiness, universal cross-organ generalization, or that every added module is uniformly beneficial.

## Review Questions

Please answer these directly:

1. Is there any remaining logic gap that would justify desk rejection or a strong major-revision decision at CMPB?
2. Are any tables, figures, or claims still unsupported by the provided strict-label result files?
3. Is the role distinction between `UL-TransXNet`, `TransXNet-MUDD+DA`, and `EfficientFormer-L1+ECA+KD` clear enough?
4. Does Figure 2 now explain the stage structure and module insertion points clearly enough for a reviewer?
5. Does Figure 5 help the mobile-deployment claim, or should the mobile evidence remain table-only?
6. Are the limitations sufficiently honest, or do they overcorrect and weaken the paper unnecessarily?
7. What are the top 5 edits or experiments that would most improve acceptance probability?

## Known Limitations to Check

- The strict package removes older calibration, CUDA trade-off, BUSI/AUL auto-ROI, and non-locked Android baseline tables because they were not regenerated under the final manifest discipline.
- The study remains retrospective and public-dataset based.
- The BUSI duplicate audit is descriptive and does not change the main benchmark denominator.
- The mobile experiment uses one Xiaomi phone and one Samsung tablet.
- Trained checkpoints, generated ROI folders, detector weights, ONNX binaries, Android packages, and full raw logs are not redistributed in this lightweight review package.

## Expected Review Style

Be concrete. Cite file names, table numbers, figure numbers, and line-level source references when possible. Separate fatal issues from polish issues.
