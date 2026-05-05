from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parent

EXTERNAL_RUN_DIRS = [
    PROJECT_ROOT / "external_validation_locked_runs" / "20260414_015615" / "tn5000_locked_eval",
    PROJECT_ROOT / "external_validation_locked_runs" / "20260414_121943" / "tn5000_3seed_locked_eval",
    PROJECT_ROOT / "external_validation_locked_runs" / "20260415_073229" / "busi_ours5fold_locked_eval",
    PROJECT_ROOT / "external_validation_locked_runs" / "20260415_073328" / "aul_ours5fold_locked_eval",
]

LODO_RUN_DIRS = [
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260414_015709",
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260414_122118",
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260415_035324",
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260415_053622",
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260415_063913",
    PROJECT_ROOT / "domain_generalization_lodo_runs" / "20260415_070603",
]

NUMERIC_FIELDS = [
    "target_acc",
    "target_bal_acc",
    "target_f1_macro",
    "target_auc",
    "target_nll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready generalization summary tables."
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "eval_reports"),
        help="Directory that will receive the timestamped summary folder.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def fmt_metric(x: float) -> str:
    return f"{x:.4f}"


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.4f} +/- {s:.4f}"


def load_external_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    recommended_variant = str(summary["recommended_variant_by_source_val"])
    target_rows = read_csv(run_dir / "target_variant_metrics.csv")
    selected_rows = [row for row in target_rows if row["variant"] == recommended_variant]
    if not selected_rows:
        raise RuntimeError(f"No rows for recommended variant under {run_dir}")
    return {
        "run_dir": run_dir,
        "source_domain": str(config["source_domain"]).lower(),
        "recommended_variant": recommended_variant,
        "rows": selected_rows,
    }


def classify_external_protocol(source_domain: str, num_ckpts_used: int) -> str:
    if source_domain == "tn5000" and num_ckpts_used == 1:
        return "DirectExternal"
    if source_domain == "tn5000" and num_ckpts_used > 1:
        return "DirectExternalEnsemble"
    return "SingleSourceMatrix"


def build_external_rows() -> list[dict]:
    rows: list[dict] = []
    for run_dir in EXTERNAL_RUN_DIRS:
        run = load_external_run(run_dir)
        for row in run["rows"]:
            num_ckpts_used = int(row["num_ckpts_used"])
            rows.append(
                {
                    "section": classify_external_protocol(run["source_domain"], num_ckpts_used),
                    "source_domain": run["source_domain"],
                    "target_domain": str(row["target_domain"]).lower(),
                    "source_domains": run["source_domain"],
                    "selection_rule": "source-val recommended variant",
                    "reported_variant": run["recommended_variant"],
                    "seed_scope": "fixed run",
                    "num_seeds": 1,
                    "num_ckpts_used": num_ckpts_used,
                    "acc_mean": fmt_metric(float(row["target_acc"])),
                    "acc_std": "",
                    "acc_report": fmt_metric(float(row["target_acc"])),
                    "bal_acc_mean": fmt_metric(float(row["target_bal_acc"])),
                    "bal_acc_std": "",
                    "bal_acc_report": fmt_metric(float(row["target_bal_acc"])),
                    "f1_macro_mean": fmt_metric(float(row["target_f1_macro"])),
                    "f1_macro_std": "",
                    "f1_macro_report": fmt_metric(float(row["target_f1_macro"])),
                    "auc_mean": fmt_metric(float(row["target_auc"])),
                    "auc_std": "",
                    "auc_report": fmt_metric(float(row["target_auc"])),
                    "nll_mean": fmt_metric(float(row["target_nll"])),
                    "nll_std": "",
                    "nll_report": fmt_metric(float(row["target_nll"])),
                    "run_dirs": str(run_dir),
                    "notes": "",
                }
            )
    return rows


def load_lodo_seed_rows() -> list[dict]:
    rows: list[dict] = []
    for run_dir in LODO_RUN_DIRS:
        rec_path = run_dir / "recommended_rows.csv"
        for row in read_csv(rec_path):
            row = dict(row)
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def summarize_counter(counter: Counter) -> str:
    return "; ".join(f"{key} x{counter[key]}" for key in sorted(counter))


def build_lodo_summary_rows(seed_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["heldout_domain"]).lower()].append(row)

    summary_rows: list[dict] = []
    for heldout_domain in sorted(grouped):
        domain_rows = sorted(grouped[heldout_domain], key=lambda row: int(row["seed"]))
        numeric = {field: [float(row[field]) for row in domain_rows] for field in NUMERIC_FIELDS}
        stats = {field: mean_std(values) for field, values in numeric.items()}
        variants = Counter(str(row["variant"]) for row in domain_rows)
        ckpt_uses = Counter(str(row["num_ckpts_used"]) for row in domain_rows)
        summary_rows.append(
            {
                "section": "LODOMultiSeedSummary",
                "source_domain": "multi_source",
                "target_domain": heldout_domain,
                "source_domains": str(domain_rows[0]["source_domains"]),
                "selection_rule": "source-val recommended per seed",
                "reported_variant": summarize_counter(variants),
                "seed_scope": "|".join(str(row["seed"]) for row in domain_rows),
                "num_seeds": len(domain_rows),
                "num_ckpts_used": summarize_counter(ckpt_uses),
                "acc_mean": fmt_metric(stats["target_acc"][0]),
                "acc_std": fmt_metric(stats["target_acc"][1]),
                "acc_report": fmt_mean_std(*stats["target_acc"]),
                "bal_acc_mean": fmt_metric(stats["target_bal_acc"][0]),
                "bal_acc_std": fmt_metric(stats["target_bal_acc"][1]),
                "bal_acc_report": fmt_mean_std(*stats["target_bal_acc"]),
                "f1_macro_mean": fmt_metric(stats["target_f1_macro"][0]),
                "f1_macro_std": fmt_metric(stats["target_f1_macro"][1]),
                "f1_macro_report": fmt_mean_std(*stats["target_f1_macro"]),
                "auc_mean": fmt_metric(stats["target_auc"][0]),
                "auc_std": fmt_metric(stats["target_auc"][1]),
                "auc_report": fmt_mean_std(*stats["target_auc"]),
                "nll_mean": fmt_metric(stats["target_nll"][0]),
                "nll_std": fmt_metric(stats["target_nll"][1]),
                "nll_report": fmt_mean_std(*stats["target_nll"]),
                "run_dirs": " | ".join(str(row["run_dir"]) for row in domain_rows),
                "notes": "",
            }
        )
    return summary_rows


def build_delta_rows(
    external_rows: list[dict], lodo_summary_rows: list[dict]
) -> list[dict]:
    direct_single = {
        row["target_domain"]: row
        for row in external_rows
        if row["section"] == "DirectExternal"
    }
    direct_ens = {
        row["target_domain"]: row
        for row in external_rows
        if row["section"] == "DirectExternalEnsemble"
    }
    delta_rows: list[dict] = []
    for row in lodo_summary_rows:
        target = row["target_domain"]
        mean_auc = float(row["auc_mean"])
        mean_bal_acc = float(row["bal_acc_mean"])
        base_single_auc = float(direct_single[target]["auc_mean"])
        base_single_bal = float(direct_single[target]["bal_acc_mean"])
        base_ens_auc = float(direct_ens[target]["auc_mean"])
        base_ens_bal = float(direct_ens[target]["bal_acc_mean"])
        delta_rows.append(
            {
                "target_domain": target,
                "lodo_auc_mean": fmt_metric(mean_auc),
                "lodo_bal_acc_mean": fmt_metric(mean_bal_acc),
                "direct_single_auc": fmt_metric(base_single_auc),
                "direct_single_bal_acc": fmt_metric(base_single_bal),
                "direct_3seed_auc": fmt_metric(base_ens_auc),
                "direct_3seed_bal_acc": fmt_metric(base_ens_bal),
                "delta_auc_vs_single": fmt_metric(mean_auc - base_single_auc),
                "delta_bal_acc_vs_single": fmt_metric(mean_bal_acc - base_single_bal),
                "delta_auc_vs_3seed": fmt_metric(mean_auc - base_ens_auc),
                "delta_bal_acc_vs_3seed": fmt_metric(mean_bal_acc - base_ens_bal),
            }
        )
    return sorted(delta_rows, key=lambda row: row["target_domain"])


def build_markdown(
    external_rows: list[dict],
    lodo_summary_rows: list[dict],
    delta_rows: list[dict],
    output_dir: Path,
) -> str:
    direct_rows = [row for row in external_rows if row["section"] in {"DirectExternal", "DirectExternalEnsemble"}]
    source_matrix_rows = [row for row in external_rows if row["section"] == "SingleSourceMatrix"]

    lines: list[str] = []
    lines.append("# Generalization Paper Summary")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"Output dir: `{output_dir}`")
    lines.append("")

    lines.append("## Direct External Baselines")
    lines.append("")
    lines.append("| Source | Target | Protocol | Variant | AUC | BalAcc | F1-macro | NLL |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(direct_rows, key=lambda r: (r["source_domain"], r["target_domain"], r["section"])):
        protocol = "1-ckpt" if row["section"] == "DirectExternal" else "3-seed ensemble"
        lines.append(
            f"| {row['source_domain']} | {row['target_domain']} | {protocol} | {row['reported_variant']} | "
            f"{row['auc_report']} | {row['bal_acc_report']} | {row['f1_macro_report']} | {row['nll_report']} |"
        )
    lines.append("")

    lines.append("## LODO 3-Seed Summary")
    lines.append("")
    lines.append("| Held-out Target | Source Domains | Variant Votes | Seeds | AUC (mean+/-std) | BalAcc (mean+/-std) | F1-macro (mean+/-std) | NLL (mean+/-std) |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(lodo_summary_rows, key=lambda r: r["target_domain"]):
        lines.append(
            f"| {row['target_domain']} | {row['source_domains']} | {row['reported_variant']} | {row['seed_scope']} | "
            f"{row['auc_report']} | {row['bal_acc_report']} | {row['f1_macro_report']} | {row['nll_report']} |"
        )
    lines.append("")

    lines.append("## LODO vs Direct TN5000 Baselines")
    lines.append("")
    lines.append("| Target | DeltaAUC vs TN5000 1-ckpt | DeltaAUC vs TN5000 3-seed | DeltaBalAcc vs TN5000 1-ckpt | DeltaBalAcc vs TN5000 3-seed |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in delta_rows:
        lines.append(
            f"| {row['target_domain']} | {row['delta_auc_vs_single']} | {row['delta_auc_vs_3seed']} | "
            f"{row['delta_bal_acc_vs_single']} | {row['delta_bal_acc_vs_3seed']} |"
        )
    lines.append("")

    lines.append("## Single-Source Cross-Domain Matrix")
    lines.append("")
    lines.append("| Source | Target | Variant | AUC | BalAcc | F1-macro | NLL |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in sorted(source_matrix_rows, key=lambda r: (r["source_domain"], r["target_domain"])):
        lines.append(
            f"| {row['source_domain']} | {row['target_domain']} | {row['reported_variant']} | "
            f"{row['auc_report']} | {row['bal_acc_report']} | {row['f1_macro_report']} | {row['nll_report']} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Direct external baseline rows are the source-val recommended variant from each run.")
    lines.append("- LODO rows report mean+/-std over seeds `17,27,37` using the per-seed source-val recommended variant.")
    lines.append("- The summary is intended for paper drafting; raw per-seed rows are preserved in `lodo_seed_rows.csv`.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"generalization_paper_summary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    external_rows = build_external_rows()
    lodo_seed_rows = load_lodo_seed_rows()
    lodo_summary_rows = build_lodo_summary_rows(lodo_seed_rows)
    delta_rows = build_delta_rows(external_rows, lodo_summary_rows)
    total_table_rows = external_rows + lodo_summary_rows
    total_table_rows.sort(key=lambda row: (row["section"], row["source_domain"], row["target_domain"]))

    write_csv(output_dir / "paper_generalization_total_table.csv", total_table_rows)
    write_csv(output_dir / "paper_generalization_lodo_seed_rows.csv", lodo_seed_rows)
    write_csv(output_dir / "paper_generalization_delta_table.csv", delta_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "external_run_dirs": [str(path) for path in EXTERNAL_RUN_DIRS],
        "lodo_run_dirs": [str(path) for path in LODO_RUN_DIRS],
        "outputs": {
            "paper_generalization_total_table.csv": str(output_dir / "paper_generalization_total_table.csv"),
            "paper_generalization_lodo_seed_rows.csv": str(output_dir / "paper_generalization_lodo_seed_rows.csv"),
            "paper_generalization_delta_table.csv": str(output_dir / "paper_generalization_delta_table.csv"),
            "README_summary.md": str(output_dir / "README_summary.md"),
        },
    }
    write_text(output_dir / "summary_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    readme = build_markdown(external_rows, lodo_summary_rows, delta_rows, output_dir)
    write_text(output_dir / "README_summary.md", readme)
    write_text(PROJECT_ROOT / "eval_reports" / "generalization_paper_summary_latest.txt", str(output_dir))

    print(f"[OK] Summary written to: {output_dir}")


if __name__ == "__main__":
    main()
