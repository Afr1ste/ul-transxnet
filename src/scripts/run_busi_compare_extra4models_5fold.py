from __future__ import annotations

import argparse
from pathlib import Path

import run_busi_compare_5models_5fold as base


PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_PTR = PROJECT_ROOT / "eval_reports" / "busi_compare_extra4models_latest.txt"

EXTRA4_CONFIGS = [
    dict(
        name="DENSENET121_autoCW_exp030",
        display_name="DenseNet121",
        model_family="timm",
        backbone_name="densenet121",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **dict(base.COMMON),
    ),
    dict(
        name="CONVNEXTT_autoCW_exp030",
        display_name="ConvNeXt-Tiny",
        model_family="timm",
        backbone_name="convnext_tiny",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **dict(base.COMMON),
    ),
    dict(
        name="REPVITM11_autoCW_exp030",
        display_name="RepViT-M1.1",
        model_family="timm",
        backbone_name="repvit_m1_1",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **dict(base.COMMON, use_ema=False),
    ),
    dict(
        name="EFFFORMERL1_autoCW_exp030",
        display_name="EfficientFormer-L1",
        model_family="timm",
        backbone_name="efficientformer_l1",
        backbone_module="",
        backbone_func="",
        backbone_out_dim=0,
        **dict(base.COMMON, input_size=224),
    ),
]


def select_configs(only_configs: str):
    if not str(only_configs).strip():
        return [dict(cfg) for cfg in EXTRA4_CONFIGS]
    allow = {x.strip() for x in str(only_configs).split(",") if x.strip()}
    selected = [
        dict(cfg)
        for cfg in EXTRA4_CONFIGS
        if cfg["name"] in allow or cfg["display_name"] in allow
    ]
    if not selected:
        raise ValueError(f"No extra4 configs matched: {only_configs}")
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BUSI extra-4 baseline comparison on 5 folds."
    )
    p.add_argument("--output-root", default="busi_roi_runs_compare_extra4models_5fold")
    p.add_argument("--log-root", default="busi_compare_extra4models_5fold_logs")
    p.add_argument("--skip-if-complete", type=int, default=1)
    p.add_argument("--continue-on-error", type=int, default=0)
    p.add_argument("--only-configs", default="")
    return p.parse_args()


def latest_log_dir(log_root: str) -> Path | None:
    root = PROJECT_ROOT / log_root
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def is_complete(log_dir: Path, expected_runs: int) -> bool:
    required = [
        log_dir / "batch_status.csv",
        log_dir / "aggregate_by_config.csv",
        log_dir / "model_level_summary.csv",
        log_dir / "paper_main_table_with_ensemble.csv",
    ]
    if not all(p.exists() for p in required):
        return False
    lines = (log_dir / "batch_status.csv").read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != expected_runs + 1:
        return False
    return True


def write_latest_ptr(log_dir: Path) -> None:
    LATEST_PTR.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PTR.write_text(str(log_dir), encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected_configs = select_configs(args.only_configs)
    expected_runs = len(selected_configs) * len(base.FOLDS)

    if bool(args.skip_if_complete):
        prev = latest_log_dir(args.log_root)
        if prev is not None and is_complete(prev, expected_runs):
            write_latest_ptr(prev)
            print(f"[SKIP] Existing complete BUSI extra4 compare found: {prev}")
            return 0

    base.OUTPUT_ROOT = args.output_root
    base.LOG_ROOT = args.log_root
    base.CONTINUE_ON_ERROR = bool(args.continue_on_error)
    base.BASE_CONFIGS = selected_configs
    base.EXPERIMENTS = base.build_experiments()

    print("=" * 100)
    print("BUSI extra4 comparison")
    print(f"output_root = {base.OUTPUT_ROOT}")
    print(f"log_root    = {base.LOG_ROOT}")
    print(f"folds       = {base.FOLDS}")
    print(f"models      = {[cfg['display_name'] for cfg in base.BASE_CONFIGS]}")
    print("=" * 100)

    rc = base.main()

    log_dir = latest_log_dir(args.log_root)
    if log_dir is not None:
        write_latest_ptr(log_dir)
        print(f"[DONE] latest_log_dir = {log_dir}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
