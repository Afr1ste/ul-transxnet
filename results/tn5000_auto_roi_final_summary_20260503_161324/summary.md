# TN5000 automatic ROI experiment final summary

## Detector localization

- Test mAP50: 0.9520 +/- 0.0090
- Test mAP50-95: 0.6373 +/- 0.0080
- Test mean IoU: 0.7855 +/- 0.0033
- Test recall@IoU0.75: 0.7880 +/- 0.0122
- No-box rate: 0.0000 across all three seeds

## Closed-loop classification with detector ROI

- Test AUC: 0.9437 +/- 0.0042
- Test balanced accuracy: 0.8686 +/- 0.0056
- Test macro F1: 0.8519 +/- 0.0116
- Test accuracy: 0.8820 +/- 0.0115

## Crop-rule reading

- The current automatic ROI protocol is `auto + rect + expand=0.3`.
- In the seed-27 sweep, `auto + rect + expand=0.4` gives the highest automatic AUC, while `auto + rect + expand=0.3` gives the highest automatic balanced accuracy / macro-F1 among the compact rows.
- Oracle boxes still outperform detected boxes, so the closed-loop gap is mainly an ROI localization / crop protocol gap rather than a classifier-only issue.

## Robustness probe reading

- Light bbox jitter and predicted-box mix did not improve the automatic-ROI operating point enough to replace the original classifier.
- Keep these as negative probes or appendix diagnostics, not as the main method.

## Source directories

- Detector 3-seed summary: `<LOCAL_THYROID_ROOT>\eval_reports\tn5000_auto_roi_detector_3seed_summary_20260502_130557`
- Crop rule sweep: `<LOCAL_THYROID_ROOT>\eval_reports\tn5000_roi_crop_rule_sweep_s27_20260502_141635`
- Light bbox jitter eval: `<LOCAL_THYROID_ROOT>\eval_reports\tn5000_auto_roi_bboxjitter_light_20260502_185435`
- Predicted-box mix eval: `<LOCAL_THYROID_ROOT>\eval_reports\tn5000_auto_roi_predboxmix_20260502_185435`
