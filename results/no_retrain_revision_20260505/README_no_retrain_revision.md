# No-retrain revision tables

Generated at: 2026-05-05T19:33:26

These files were generated from existing completed-result CSVs only. No images, manifests, checkpoints, or training entrypoints were loaded. This is intentional because the current intermediate dataset state may contain label drift relative to the frozen result logs.

## Key validation-selection audit

| Variant | TN5000 val AUC | BUSI val AUC | AUL val AUC | Mean val AUC | Selected |
|---|---:|---:|---:|---:|---|
| TransXNet | 0.9577 | 0.7796 | 0.7464 | 0.8279 | no |
| TransXNet-GG | 0.9632 | 0.8011 | 0.7390 | 0.8345 | no |
| GGG-noMCA | 0.9700 | 0.8957 | 0.8798 | 0.9152 | no |
| UL-TransXNet | 0.9638 | 0.9199 | 0.8827 | 0.9222 | yes |

## Sources

- TransXNet / TN5000: `tn5000_p0_structure_current_3seed_logs/20260429_030247/all_runs_metrics.csv` display_name_cfg=TransXNet
- TransXNet / BUSI: `busi_p0_structure_clean_5fold_logs/20260425_034843/all_runs_metrics.csv` display_name_cfg=TransXNet
- TransXNet / AUL: `aul_p0_structure_clean_5fold_logs/20260425_030033/all_runs_metrics.csv` display_name_cfg=TransXNet
- TransXNet-GG / TN5000: `tn5000_p0_structure_current_3seed_logs/20260429_104255/all_runs_metrics.csv` display_name_cfg=TransXNet-GG
- TransXNet-GG / BUSI: `busi_p0_structure_clean_5fold_logs/20260425_034843/all_runs_metrics.csv` display_name_cfg=TransXNet-GG
- TransXNet-GG / AUL: `aul_p0_structure_clean_5fold_logs/20260425_030033/all_runs_metrics.csv` display_name_cfg=TransXNet-GG
- GGG-noMCA / TN5000: `tn5000_ggg_nomca_current_3seed_logs/20260427_051327/all_runs_metrics.csv`
- GGG-noMCA / BUSI: `busi_ggg_nomca_clean_5fold_safe_logs/20260427_123519/all_runs_metrics.csv`
- GGG-noMCA / AUL: `aul_ggg_nomca_clean_5fold_safe_logs/20260427_140214/all_runs_metrics.csv`
- UL-TransXNet / TN5000: `tn5000_ggg_mca_enabled_3seed_logs/20260426_093728/all_runs_metrics.csv`
- UL-TransXNet / BUSI: `busi_ggg_mca_clean_5fold_safe_logs/20260426_165332/all_runs_metrics.csv`
- UL-TransXNet / AUL: `aul_ggg_mca_clean_5fold_safe_logs/20260426_200618/all_runs_metrics.csv`