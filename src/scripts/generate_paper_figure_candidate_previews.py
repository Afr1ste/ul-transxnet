from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import generate_paper_auto_roi_workflow_figure as auto_fig
import generate_paper_gradcam_sanity_figure as grad_fig


ROOT = Path(r"<LOCAL_THYROID_ROOT>")
PAPER_FIG_DIR = Path(r"<LOCAL_MANUSCRIPT_ROOT>\figures")
OUT_DIR = PAPER_FIG_DIR / "candidate_previews"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def select_auto_roi_candidates(dataset: str, cfg: dict[str, Any], n: int = 4) -> pd.DataFrame:
    box_df = pd.read_csv(auto_fig.BOX_ROOT / "box_quality_per_image.csv")
    pred_df = pd.read_csv(auto_fig.CLOSED_ROOT / dataset / "auto" / f"fold{cfg['fold']}" / "test_predictions.csv")
    sub = box_df[
        (box_df["dataset"] == dataset)
        & (box_df["fold"].astype(int) == int(cfg["fold"]))
        & (box_df["split"] == "test")
    ].copy()
    sub = sub.merge(
        pred_df[["image_id", "true_label", "pred_label", "prob_class1", "threshold", "is_wrong"]],
        on="image_id",
        how="inner",
    )
    sub = sub[(sub["no_detection"].astype(int) == 0) & (sub["is_wrong"].astype(int) == 0)].copy()
    sub["margin"] = np.abs(sub["prob_class1"].astype(float) - sub["threshold"].astype(float))
    sub["pred_area"] = (
        (sub["pred_xmax"].astype(float) - sub["pred_xmin"].astype(float))
        * (sub["pred_ymax"].astype(float) - sub["pred_ymin"].astype(float))
    )
    sub = sub[(sub["margin"] > 0.035) & (sub["pred_area"] > 8000)].copy()

    # Prefer readable, representative boxes: good IoU, confident classification,
    # non-tiny lesion area, and a small penalty for overly perfect examples.
    iou = sub["iou"].astype(float)
    margin = sub["margin"].astype(float)
    area = sub["pred_area"].astype(float)
    area_norm = np.clip((area - area.quantile(0.10)) / (area.quantile(0.90) - area.quantile(0.10) + 1e-8), 0, 1)
    perfection_penalty = np.clip(iou - 0.88, 0, None) * 0.35
    sub["display_score"] = 0.58 * iou + 0.24 * np.clip(margin, 0, 0.4) + 0.18 * area_norm - perfection_penalty
    return sub.sort_values("display_score", ascending=False).head(n).reset_index(drop=True)


def save_auto_roi_sheet(dataset: str, cfg: dict[str, Any], candidates: pd.DataFrame) -> Path:
    label = cfg["label"]
    fig, axes = plt.subplots(len(candidates), 4, figsize=(10.7, 2.42 * len(candidates)), dpi=220)
    if len(candidates) == 1:
        axes = axes[None, :]
    col_titles = ["Original", "Detector output", "Expanded ROI", "Classifier input"]

    for r, row in candidates.iterrows():
        row = row.to_dict()
        img = Image.open(row["image_path"]).convert("L")
        w, h = img.size
        gt = auto_fig.to_float_box(row, "gt")
        pred = auto_fig.to_float_box(row, "pred")
        sq = auto_fig.expanded_square(pred, w, h, cfg["expand"])
        crop = auto_fig.crop_square(img, sq)
        panels = [
            img,
            auto_fig.draw_overlay(img, gt_box=gt, pred_box=pred),
            auto_fig.draw_overlay(img, pred_box=pred, sq_box=sq),
            crop,
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#202020")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight="semibold")
        axes[r, 0].set_ylabel(f"{label}-{chr(65 + r)}", fontsize=10, fontweight="bold")
        axes[r, 1].text(
            0.5,
            -0.10,
            f"id={row['image_id']} | IoU={float(row['iou']):.2f}",
            transform=axes[r, 1].transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
        )
        axes[r, 3].text(
            0.5,
            -0.10,
            f"p1={float(row['prob_class1']):.2f}, thr={float(row['threshold']):.2f}",
            transform=axes[r, 3].transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
        )
    fig.suptitle(f"Figure 4 candidates: {label}", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.965])
    out = OUT_DIR / f"fig4_auto_roi_candidates_{dataset}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def iter_gradcam_predictions(case: grad_fig.GradCamCase, limit: int = 45) -> list[dict[str, str]]:
    rows = grad_fig.read_predictions(case.run_dir / "test_predictions.csv")
    correct = [r for r in rows if r.get("is_wrong") == "0"]
    positives = [
        r
        for r in correct
        if r.get("true_label") == r.get("pred_label")
        and _safe_float(r.get("prob_class1")) >= _safe_float(r.get("threshold"), 0.5)
    ]
    pool = positives if positives else correct
    pool = sorted(pool, key=lambda r: max(_safe_float(r.get("prob_class1")), 1.0 - _safe_float(r.get("prob_class1"))), reverse=True)
    return pool[:limit]


def gradcam_candidates_for_case(case: grad_fig.GradCamCase, device: torch.device, n: int = 4) -> list[dict[str, Any]]:
    case = replace(case, preferred_image_id=None)
    module = grad_fig.import_train_module(case.module_name)
    dataset = grad_fig.make_dataset(module, case)
    _train_transform, eval_transform = module.build_transforms()
    model = grad_fig.instantiate_model(module, case.run_dir / case.checkpoint_name, device)
    target_layer = grad_fig.get_target_layer(model)
    gradcam = grad_fig.GradCAM(model, target_layer)
    items: list[dict[str, Any]] = []
    try:
        for pred in iter_gradcam_predictions(case):
            try:
                roi_img, label, image_id = grad_fig.get_roi_by_id(dataset, pred["image_id"])
            except KeyError:
                continue
            tensor = eval_transform(roi_img).unsqueeze(0).to(device)
            target_class = int(label)
            cam, probs = gradcam(tensor, target_class)
            rgb = grad_fig.resize_for_display(roi_img, size=256)
            overlay = grad_fig.cam_overlay(rgb, cam)
            conf_true = float(probs[target_class])
            center = grad_fig.center_energy(cam)
            white = float((rgb.mean(axis=2) > 0.90).mean())
            peakiness = float(np.percentile(cam, 95) - np.mean(cam))
            # Display score favors centered, concentrated heatmaps with enough
            # confidence, while penalizing blank borders and overly diffuse maps.
            score = 0.50 * center + 0.24 * conf_true + 0.20 * peakiness - 0.75 * white
            items.append(
                {
                    "dataset": case.dataset,
                    "image_id": image_id,
                    "true_label": int(label),
                    "pred_label": int(pred["pred_label"]),
                    "prob_class1": _safe_float(pred.get("prob_class1")),
                    "threshold": _safe_float(pred.get("threshold"), 0.5),
                    "prob_true_single_ckpt": conf_true,
                    "center_energy": center,
                    "peakiness": peakiness,
                    "white_fraction": white,
                    "score": score,
                    "rgb": rgb,
                    "overlay": overlay,
                }
            )
    finally:
        gradcam.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sorted(items, key=lambda x: x["score"], reverse=True)[:n]


def save_gradcam_sheet(dataset: str, candidates: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, len(candidates), figsize=(2.45 * len(candidates), 4.65), dpi=220)
    if len(candidates) == 1:
        axes = axes[:, None]
    for c, item in enumerate(candidates):
        axes[0, c].imshow(item["rgb"])
        axes[1, c].imshow(item["overlay"])
        axes[0, c].set_title(
            f"{chr(65 + c)}: {item['image_id']}\nconf={item['prob_true_single_ckpt']:.2f}, center={item['center_energy']:.2f}",
            fontsize=8.2,
        )
        for r in range(2):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_edgecolor("#202020")
    axes[0, 0].set_ylabel("ROI input", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9, fontweight="bold")
    fig.suptitle(f"Figure 5 candidates: {dataset}", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out = OUT_DIR / f"fig5_gradcam_candidates_{dataset.lower()}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
        }
    )

    summary_lines: list[str] = []
    for ds, cfg in auto_fig.DATASETS.items():
        candidates = select_auto_roi_candidates(ds, cfg, n=4)
        out = save_auto_roi_sheet(ds, cfg, candidates)
        summary_lines.append(f"FIG4 {cfg['label']} -> {out}")
        for i, row in candidates.iterrows():
            summary_lines.append(
                f"  {chr(65+i)} id={row['image_id']} iou={float(row['iou']):.4f} "
                f"p1={float(row['prob_class1']):.4f} thr={float(row['threshold']):.4f} "
                f"score={float(row['display_score']):.4f}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_lines.append(f"Grad-CAM device={device}")
    for case in grad_fig.CASES:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            candidates = gradcam_candidates_for_case(case, device, n=4)
        out = save_gradcam_sheet(case.dataset, candidates)
        summary_lines.append(f"FIG5 {case.dataset} -> {out}")
        for i, item in enumerate(candidates):
            summary_lines.append(
                f"  {chr(65+i)} id={item['image_id']} true={item['true_label']} pred={item['pred_label']} "
                f"p1={item['prob_class1']:.4f} conf_true={item['prob_true_single_ckpt']:.4f} "
                f"center={item['center_energy']:.4f} peak={item['peakiness']:.4f} score={item['score']:.4f}"
            )

    summary_path = OUT_DIR / "candidate_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(summary_path)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
