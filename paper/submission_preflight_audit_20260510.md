# Submission Preflight Audit - 2026-05-10

## Verdict

The manuscript is now in a mature pre-submission state for Computer Methods and Programs in Biomedicine as an Original Research article, subject to final author-side administrative checks and the manifest/prediction artifact gate. I did not find a severe writing-level logic flaw that should block advisor review.

The paper is no longer framed as a new universal medical vision backbone. It is framed as a bounded computing-methods workflow for ROI-robust ultrasound lesion classification, teacher analysis, mobile distillation, and multi-organ protocol separation. This is the right level of claim for the present evidence.

## CMPB Compliance

- Ordinary CMPB Original Research submission is open through Editorial Manager; no near-term fixed deadline was found for regular submissions.
- Main manuscript uses Elsevier `elsarticle` source and compiles to PDF.
- Structured abstract is present and below the CMPB 350-word limit.
- Keywords: 6.
- Highlights: 5 bullets, each below the 85-character Elsevier highlight limit.
- Sections follow Introduction, Methods, Results, Discussion, declarations, and references.
- Cover letter is prepared with CMPB fit, known background, added contribution, and non-duplicate-submission confirmation.
- Title-page file is prepared, but the corresponding-author phone number still needs to be filled.
- CRediT, ethics, funding, competing interest, generative-AI declaration, and data availability statements are included.

## Scientific Logic Check

- Cross-organ wording is now controlled: the manuscript uses multi-organ public benchmark evaluation and explicitly states that LODO transfer remains weak.
- Mobile wording is controlled: the TransXNet-family teacher is not claimed as the mobile solution; EfficientFormer distillation is presented as the practical mobile path.
- ROI dependence is controlled: oracle ROI is treated as a classifier protocol, with detector ROI and box-noise experiments used as robustness evidence rather than full clinical deployment proof.
- The TransXNet comparison is transparent: the paper reports that UL-TransXNet is not a TN5000 AUC improvement over original TransXNet, while showing stronger BUSI/AUL behavior.
- The ablation story is no longer monotonic stacking: TransXNet-MUDD+DA is identified as the strongest TN5000 robustness variant, and MCA is described as dataset-dependent.
- The frozen analysis-label snapshot is stated separately from descriptive dataset counts, reducing the risk that Table 1 appears internally inconsistent.

## Remaining Risks

- The label-snapshot issue remains the main submission risk. The paper now states a manifest-based provenance chain, but the actual release package must contain the split/protocol manifests and per-case prediction CSVs described in `artifact_manifest_requirements_cmpb.md`.
- The 1000-D task-projection head is now explained, but a reviewer could still ask for a head ablation. This is not a submission blocker for CMPB.
- Mobile results are one-device evidence. The manuscript correctly calls this feasibility/prototype evidence, not clinical deployment.
- LODO cross-organ performance is weak. This is acceptable because the paper treats it as a boundary result rather than a positive claim.
- Figures are serviceable and clear, but not top-conference polished. The paper looks like an Elsevier review manuscript, not a CVPR/MICCAI camera-ready paper.

## Visual and Formatting Check

- Main manuscript: 17 pages, 7 tables, 6 figures.
- Supplement: 8 tables, 6 figures.
- Key pages with architecture, benchmark tables, ROI robustness, mobile latency, cross-organ heatmap, discussion, and declarations were rendered and visually inspected.
- No overfull boxes, undefined citations, undefined references, or LaTeX errors were found after recompilation.
- One minor underfull line remains in the contribution list; this is cosmetic and not submission-blocking.

## Top-Tier Gap

Compared with a benchmark top-conference paper, the main gap is not basic correctness anymore. The gap is contribution sharpness and visual polish:

- A top paper would usually have a more original model mechanism, not mainly a carefully controlled reorganization and evaluation.
- A top paper would likely include stronger external validation, prospective or institution-held-out data, and better multi-device deployment.
- A top paper would use more polished, unified visual design and a stronger first figure.
- This manuscript is better matched to a biomedical computing journal than to a top AI conference.

## Practical Readiness Score

Current manuscript-writing readiness: approximately 77-80/100 for the intended SCI2/CMPB target.

This is in a reasonable submission band for advisor review. It is not an 80+/100 top-tier version, mainly because external validation and visual/conceptual novelty remain limited. For CMPB, the current version is defensible only if the author-side upload fields are completed and the manifest/prediction artifact chain is actually assembled.

## Before Upload

- Fill corresponding-author phone in `cover_letter_cmpb.md` and `title_page_cmpb.md`.
- Confirm author order, affiliation wording, and email.
- Decide whether the supplement is uploaded as a separate supplementary file.
- Confirm repository URL and data-sharing wording.
- Assemble the manifest and prediction-CSV artifact chain before final upload.
- Refresh the source archive after any manual author edits.
