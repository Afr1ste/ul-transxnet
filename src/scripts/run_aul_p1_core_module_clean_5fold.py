from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_aul_roi_binary_compare_5models_5fold as base


PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_PTR = PROJECT_ROOT / "eval_reports" / "aul_p1_core_module_clean_latest.txt"

BACKBONES = [
    {
        "key": "base",
        "display_name": "GGG-core",
        "backbone_name": "transxnet_t",
        "backbone_module": "models.transxnetggg_nomca_base",
        "backbone_func": "transxnet_t",
        "backbone_out_dim": 1000,
    },
    {
        "key": "mudd",
        "display_name": "GGG-core+MUDD",
        "backbone_name": "transxnet_t",
        "backbone_module": "models.transxnetggg_nomca_mudd",
        "backbone_func": "transxnet_t",
        "backbone_out_dim": 1000,
    },
    {
        "key": "da",
        "display_name": "GGG-core+DA",
        "backbone_name": "transxnet_t",
        "backbone_module": "models.transxnetggg_nomca_da",
        "backbone_func": "transxnet_t",
        "backbone_out_dim": 1000,
    },
    {
        "key": "mudd_da",
        "display_name": "GGG-core+MUDD+DA",
        "backbone_name": "transxnet_t",
        "backbone_module": "models.transxnetggg_nomca",
        "backbone_func": "transxnet_t",
        "backbone_out_dim": 1000,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AUL clean P1 core module ablation.")
    p.add_argument("--output-root", default="aul_roi_runs_p1_core_module_clean_5fold")
    p.add_argument("--log-root", default="aul_p1_core_module_clean_5fold_logs")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument(
        "--only-backbones",
        default="",
        help="Comma-separated subset of: base,mudd,da,mudd_da",
    )
    p.add_argument("--continue-on-error", type=int, default=0)
    p.add_argument("--skip-if-complete", type=int, default=1)
    return p.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def resolve_backbones(raw: str) -> list[dict]:
    if not str(raw).strip():
        return list(BACKBONES)
    allow = {x.strip().lower() for x in str(raw).split(",") if x.strip()}
    selected = [cfg for cfg in BACKBONES if cfg["key"] in allow]
    if not selected:
        raise ValueError(f"No valid P1 AUL backbones selected from: {raw}")
    return selected


def latest_log_dir(log_root: str) -> Path | None:
    root = PROJECT_ROOT / log_root
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else None


def is_complete(log_dir: Path, expected_runs: int) -> bool:
    batch_csv = log_dir / "batch_status.csv"
    summary_csv = log_dir / "paper_main_table_with_ensemble.csv"
    if not batch_csv.exists() or not summary_csv.exists():
        return False
    lines = batch_csv.read_text(encoding="utf-8-sig").splitlines()
    return len(lines) == expected_runs + 1


def write_latest_ptr(log_dir: Path) -> None:
    LATEST_PTR.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PTR.write_text(str(log_dir), encoding="utf-8")


def main() -> int:
    args = parse_args()
    folds = parse_int_list(args.folds)
    selected = resolve_backbones(args.only_backbones)
    expected_runs = len(folds) * len(selected)

    if bool(args.skip_if_complete):
        prev = latest_log_dir(args.log_root)
        if prev is not None and is_complete(prev, expected_runs):
            write_latest_ptr(prev)
            print(f"[SKIP] Existing complete AUL P1 run found: {prev}")
            return 0

    common = dict(base.COMMON)
    base_configs = []
    for cfg in selected:
        item = dict(common)
        item.update(
            {
                "name": f"{cfg['key'].upper()}_aul_p1_clean",
                "display_name": cfg["display_name"],
                "model_family": "custom",
                "backbone_name": cfg["backbone_name"],
                "backbone_module": cfg["backbone_module"],
                "backbone_func": cfg["backbone_func"],
                "backbone_out_dim": cfg["backbone_out_dim"],
            }
        )
        base_configs.append(item)

    base.OUTPUT_ROOT = args.output_root
    base.LOG_ROOT = args.log_root
    base.GLOBAL_SEED = int(args.seed)
    base.FOLDS = folds
    base.CONTINUE_ON_ERROR = bool(args.continue_on_error)
    base.BASE_CONFIGS = base_configs
    base.EXPERIMENTS = base.build_experiments()

    print("=" * 100)
    print("AUL P1 clean core module ablation")
    print(f"output_root = {base.OUTPUT_ROOT}")
    print(f"log_root    = {base.LOG_ROOT}")
    print(f"seed        = {base.GLOBAL_SEED}")
    print(f"folds       = {base.FOLDS}")
    print(f"backbones   = {[cfg['key'] for cfg in selected]}")
    print("=" * 100)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(PROJECT_ROOT / "run_aul_roi_binary_compare_5models_5fold.py"),
            "--output-root",
            base.OUTPUT_ROOT,
            "--log-root",
            base.LOG_ROOT,
            "--only-configs",
            ",".join(cfg["name"] for cfg in base_configs),
            "--continue-on-error",
            "1" if base.CONTINUE_ON_ERROR else "0",
        ]
        code = int(base.main())
    finally:
        sys.argv = old_argv

    log_dir = latest_log_dir(args.log_root)
    if log_dir is not None:
        write_latest_ptr(log_dir)
        print(f"[DONE] latest_log_dir = {log_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
