# CMPB Submission Checklist

## Ready

- Regular CMPB submission has no near-term fixed deadline.
- A CMPB-targeted `elsarticle` manuscript draft is prepared in `main.tex`.
- Structured abstract is included.
- Highlights are prepared in `highlights_cmpb.txt`.
- Cover letter is prepared in `cover_letter_cmpb.md`.
- Separate title page is prepared in `title_page_cmpb.md`.
- Figures used by the manuscript are copied into this directory, same level as `main.tex`.
- `refs.bib` is copied into this directory.
- Declarations are included: CRediT, ethics, funding, competing interest, generative AI, and data availability.
- Manifest/prediction artifact requirements are documented in `artifact_manifest_requirements_cmpb.md`.
- A flat source package can be built with `python scripts/build_cmpb_submission_package.py`.

## Before actual upload

- Fill the corresponding author's phone number in the cover letter.
- Fill the corresponding author's phone number in the title page.
- Review the included `declaration_of_competing_interest.docx`; replace it with the official Elsevier-generated declaration if Editorial Manager requires that exact output.
- Confirm all author names, order, affiliations, and email addresses.
- Confirm whether the public repository URL should be `https://github.com/Afr1ste/ul-transxnet` or another address.
- Decide whether to submit supplementary figures/tables as a separate file.
- Assemble and verify the file-level manifests, hashes, and per-case prediction CSVs required by `artifact_manifest_requirements_cmpb.md`.
- Compile `main.tex` cleanly and inspect the PDF.
- Check that every figure/table cited in text appears in the PDF.
- Check that every cited reference appears in `refs.bib` and every important reference has DOI/metadata if available.
- If using Editorial Manager LaTeX source upload, flatten all required files into one root folder; do not rely on nested figure folders.
