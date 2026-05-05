#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run TN5000 3-seed verification for the patched TransXNet-GGG with MCA active."""

import run_tn5000_compare_5models_3seed as runner


runner.OUTPUT_ROOT = "tn5000_roi_runs_ggg_mca_enabled_3seed"
runner.LOG_ROOT = "tn5000_ggg_mca_enabled_3seed_logs"
runner.CONTINUE_ON_ERROR = False

runner.BASE_CONFIGS = [
    dict(
        name="OURS_GGG_MCAON_autoCW_exp030",
        display_name="Ours-GGG-MCAON",
        model_family="custom",
        backbone_name="transxnet_t",
        backbone_module="models.transxnetggg",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
        **runner.COMMON,
    )
]
runner.EXPERIMENTS = runner.build_experiments()


if __name__ == "__main__":
    raise SystemExit(runner.main())
