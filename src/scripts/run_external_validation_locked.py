from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import fl_tn5000_roi_train_unified as core
import domain_generalization_utils as dgu


DEFAULT_TN5000_MAINLINE_CKPT = (
    Path(__file__).resolve().parent
    / "tn5000_roi_runs_E06_mainline_multiseed"
    / "20260329_024552"
    / "epoch062_bal_acc_0.9208.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked external validation with source-val-only calibration."
    )
    parser.add_argument(
        "--source-domain",
        default="tn5000",
        choices=sorted(dgu.DOMAIN_SPECS.keys()),
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="One or more checkpoint paths. If omitted, the TN5000 mainline checkpoint is used.",
    )
    parser.add_argument(
        "--target-domains",
        nargs="+",
        default=["tn5000", "busi", "aul"],
        choices=sorted(dgu.DOMAIN_SPECS.keys()),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-root", default="external_validation_locked_runs")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[str]:
    if args.checkpoint_paths:
        return [str(Path(path).resolve()) for path in args.checkpoint_paths]
    if args.source_domain == "tn5000":
        return [str(DEFAULT_TN5000_MAINLINE_CKPT.resolve())]
    raise ValueError("checkpoint paths are required when source_domain is not tn5000")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{args.source_domain}_locked_eval"
    output_root = Path(args.output_root) / timestamp
    result_dir = output_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = resolve_checkpoint_paths(args)
    dgu.configure_core_mainline(
        output_root=output_root,
        run_name=run_name,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=1,
        num_workers=args.num_workers,
    )

    _, eval_transform = core.build_transforms()
    source_val_dataset = dgu.build_domain_dataset(args.source_domain, "val", transform=eval_transform)
    source_val_loader = torch.utils.data.DataLoader(
        source_val_dataset,
        batch_size=core.Config.batch_size,
        shuffle=False,
        num_workers=core.Config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    target_loaders = {}
    for domain_name in args.target_domains:
        target_dataset = dgu.build_domain_dataset(domain_name, "test", transform=eval_transform)
        target_loaders[domain_name] = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=core.Config.batch_size,
            shuffle=False,
            num_workers=core.Config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = core.ClassificationModel(num_classes=core.Config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    eval_summary = dgu.evaluate_variants_with_source_val(
        model=model,
        candidate_ckpt_paths=ckpt_paths,
        source_val_loader=source_val_loader,
        target_loaders=target_loaders,
        criterion=criterion,
        device=device,
        result_dir=result_dir,
    )

    dgu.write_json(
        result_dir / "run_config.json",
        {
            "source_domain": args.source_domain,
            "checkpoint_paths": ckpt_paths,
            "target_domains": [str(x).lower() for x in args.target_domains],
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "core_config": core.export_config_dict(),
            "source_val_dataset": dgu.dataset_summary(source_val_dataset),
            "target_datasets": {
                str(name).lower(): dgu.dataset_summary(loader.dataset)
                for name, loader in target_loaders.items()
            },
            "recommended_variant_by_source_val": eval_summary["recommended_variant_by_source_val"],
        },
    )

    print("=" * 100)
    print("Locked external validation finished")
    print(f"[OUT ] {result_dir}")
    print(f"[RECO] {eval_summary['recommended_variant_by_source_val']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
