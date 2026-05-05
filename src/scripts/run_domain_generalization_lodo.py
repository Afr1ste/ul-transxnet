from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import fl_tn5000_roi_train_unified as core
import domain_generalization_utils as dgu


ALL_DOMAINS = ["tn5000", "busi", "aul"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-domain-out training with source-val-only model selection."
    )
    parser.add_argument(
        "--heldout-domains",
        nargs="+",
        default=["all"],
        choices=ALL_DOMAINS + ["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 27, 37])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--domain-balanced-train", type=int, default=1)
    parser.add_argument("--output-root", default="domain_generalization_lodo_runs")
    return parser.parse_args()


def normalize_heldout_domains(raw_domains: list[str]) -> list[str]:
    lowered = [str(x).strip().lower() for x in raw_domains]
    if "all" in lowered:
        return list(ALL_DOMAINS)
    return lowered


def sorted_candidate_ckpts(improved_ckpts: list[dict], best_path: Path) -> list[str]:
    rows = sorted(
        improved_ckpts,
        key=lambda row: (float(row["score"]), int(row["epoch"])),
        reverse=True,
    )
    paths = [str(Path(row["path"])) for row in rows if Path(row["path"]).exists()]
    if paths:
        return paths
    return [str(best_path)]


def main() -> None:
    args = parse_args()
    heldout_domains = normalize_heldout_domains(args.heldout_domains)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    recommended_rows = []
    all_variant_rows = []

    for heldout_domain in heldout_domains:
        source_domains = [domain for domain in ALL_DOMAINS if domain != heldout_domain]
        for seed in args.seeds:
            run_name = f"lodo_{heldout_domain}_s{seed}"
            dgu.configure_core_mainline(
                output_root=output_root,
                run_name=run_name,
                seed=seed,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                num_workers=args.num_workers,
            )

            train_transform, eval_transform = core.build_transforms()
            train_dataset = dgu.build_multi_domain_dataset(
                source_domains, "train", transform=train_transform
            )
            val_dataset = dgu.build_multi_domain_dataset(
                source_domains, "val", transform=eval_transform
            )
            test_dataset = dgu.build_domain_dataset(
                heldout_domain, "test", transform=eval_transform
            )
            dataloaders = dgu.build_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                domain_balanced_train=bool(args.domain_balanced_train),
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result_dir = core.create_result_dir()
            model = core.ClassificationModel(num_classes=core.Config.num_classes).to(device)

            class_weights = None
            if core.Config.use_class_weight:
                class_weights = core.build_class_weights(train_dataset, device)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=core.Config.label_smoothing,
            )
            optimizer = core.build_optimizer(model)

            history, best_path, last_path, best_epoch, best_score, improved_ckpts = core.train_model(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                result_dir=result_dir,
                device=device,
            )
            curves_path = core.save_curves(history, result_dir)

            eval_model = core.ClassificationModel(num_classes=core.Config.num_classes).to(device)
            eval_summary = dgu.evaluate_variants_with_source_val(
                model=eval_model,
                candidate_ckpt_paths=sorted_candidate_ckpts(improved_ckpts, best_path),
                source_val_loader=dataloaders["val"],
                target_loaders={heldout_domain: dataloaders["test"]},
                criterion=criterion,
                device=device,
                result_dir=result_dir,
            )

            recommended_variant = eval_summary["recommended_variant_by_source_val"]
            recommended_target_row = None
            for row in eval_summary["target_rows"]:
                if row["variant"] == recommended_variant and row["target_domain"] == heldout_domain:
                    recommended_target_row = row
                    break

            run_summary = {
                "heldout_domain": heldout_domain,
                "source_domains": source_domains,
                "seed": int(seed),
                "result_dir": str(result_dir),
                "best_model_path": str(best_path),
                "last_model_path": str(last_path),
                "curves_path": str(curves_path),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "recommended_variant_by_source_val": recommended_variant,
                "train_dataset": dgu.dataset_summary(train_dataset),
                "val_dataset": dgu.dataset_summary(val_dataset),
                "test_dataset": dgu.dataset_summary(test_dataset),
                "core_config": core.export_config_dict(),
                "source_val_variants": eval_summary["source_val_variants"],
                "target_rows": eval_summary["target_rows"],
            }
            dgu.write_json(result_dir / "run_summary.json", run_summary)

            for row in eval_summary["target_rows"]:
                enriched_row = {
                    "heldout_domain": heldout_domain,
                    "source_domains": "|".join(source_domains),
                    "seed": int(seed),
                    **row,
                }
                all_variant_rows.append(enriched_row)

            if recommended_target_row is not None:
                recommended_rows.append(
                    {
                        "heldout_domain": heldout_domain,
                        "source_domains": "|".join(source_domains),
                        "seed": int(seed),
                        **recommended_target_row,
                    }
                )

            dgu.write_rows_csv(output_root / "all_variant_rows.csv", all_variant_rows)
            dgu.write_rows_csv(output_root / "recommended_rows.csv", recommended_rows)

            print("=" * 100)
            print(f"[DONE] heldout={heldout_domain} seed={seed}")
            print(f"[OUT ] {result_dir}")
            print(f"[RECO] {recommended_variant}")
            print("=" * 100)


if __name__ == "__main__":
    main()
