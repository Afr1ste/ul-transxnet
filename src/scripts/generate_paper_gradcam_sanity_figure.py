from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parent
PAPER_DIR = Path(r"<LOCAL_MANUSCRIPT_ROOT>")
FIG_DIR = PAPER_DIR / "figures"
OUT_BASE = FIG_DIR / "fig_gradcam_sanity"


@dataclass(frozen=True)
class GradCamCase:
    dataset: str
    module_name: str
    dataset_class: str
    root_attr: str
    split: str
    bbox_expand: float
    run_dir: Path
    checkpoint_name: str
    preferred_image_id: str | None = None
    target_layer_key: str = "stage3_h16"


CASES = [
    GradCamCase(
        dataset="TN5000",
        module_name="fl_tn5000_roi_compare_multimodel",
        dataset_class="TN5000ROIDataset",
        root_attr="tn5000_root",
        split="test",
        bbox_expand=0.30,
        run_dir=ROOT / "tn5000_roi_runs_ggg_mca_enabled_3seed" / "20260426_121126",
        checkpoint_name="best_model_tn5000_roi.pth",
        preferred_image_id="003171",
        target_layer_key="stage3_h16",
    ),
    GradCamCase(
        dataset="BUSI",
        module_name="fl_busi_roi_compare_5fold",
        dataset_class="BUSIVOCRoiDataset",
        root_attr="data_root",
        split="test",
        bbox_expand=0.30,
        run_dir=ROOT / "busi_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_181344",
        checkpoint_name="best_model_busi_roi.pth",
        preferred_image_id="test_malignant_0206",
        target_layer_key="stage4_h32",
    ),
    GradCamCase(
        dataset="AUL",
        module_name="fl_aul_roi_binary_compare_5fold",
        dataset_class="AULBinaryVOCRoiDataset",
        root_attr="data_root",
        split="test",
        bbox_expand=0.20,
        run_dir=ROOT / "aul_roi_runs_ggg_mca_clean_5fold_safe" / "20260426_200624",
        checkpoint_name="best_model_aul_roi_bin.pth",
        preferred_image_id="malignant_000382",
        target_layer_key="stage4_h32",
    ),
]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activation: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def close(self) -> None:
        self.handle.remove()

    def _forward_hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"Grad-CAM target layer returned non-tensor output: {type(output)}")
        self.activation = output
        if output.requires_grad:
            output.retain_grad()

    def __call__(self, image_tensor: torch.Tensor, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        self.activation = None
        logits = self.model(image_tensor)
        score = logits[:, target_class].sum()
        score.backward()
        if self.activation is None:
            raise RuntimeError("Grad-CAM hook did not capture activations.")
        if self.activation.grad is None:
            raise RuntimeError("Grad-CAM target activation has no gradients.")
        activation = self.activation.detach()
        gradients = self.activation.grad.detach()
        if activation.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError(
                f"Grad-CAM expects 4D activation/gradient, got {tuple(activation.shape)} and {tuple(gradients.shape)}"
            )
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np = normalize01(cam_np)
        prob_np = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
        return cam_np, prob_np


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_candidates(case: GradCamCase, max_candidates: int = 200) -> list[dict[str, str]]:
    rows = read_predictions(case.run_dir / "test_predictions.csv")
    if case.preferred_image_id:
        for row in rows:
            if row.get("image_id") == case.preferred_image_id:
                return [row]

    correct_positive = [
        r
        for r in rows
        if r.get("true_label") == "1" and r.get("pred_label") == "1" and float(r.get("prob_class1", 0.0)) >= float(r.get("threshold", 0.5))
    ]
    if not correct_positive:
        correct_positive = [r for r in rows if r.get("is_wrong") == "0"]
    if not correct_positive:
        raise RuntimeError(f"No correct candidate found for {case.dataset}")
    return sorted(correct_positive, key=lambda r: float(r.get("prob_class1", 0.0)), reverse=True)[:max_candidates]


def import_train_module(name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.import_module(name)

    # BUSI/AUL training scripts silence custom-backbone prints under torch.no_grad().
    # That is correct for evaluation, but Grad-CAM needs gradients through the backbone.
    def silent_call_with_grad(fn, *args, **kwargs):
        fake_out = io.StringIO()
        with contextlib.redirect_stdout(fake_out), contextlib.redirect_stderr(fake_out):
            return fn(*args, **kwargs)

    if hasattr(module, "silent_call"):
        module.silent_call = silent_call_with_grad
    return module


def instantiate_model(module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    with contextlib.redirect_stdout(io.StringIO()):
        model = module.UnifiedRoiClassifier(num_classes=module.Config.num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model: torch.nn.Module, layer_key: str = "stage3_h16") -> torch.nn.Module:
    # The classifier returns a pooled vector, so Grad-CAM must hook a spatial
    # block inside the backbone. H/16 is less coarse for small ROIs; H/32 can be
    # cleaner when the H/16 maps are overly diffuse.
    backbone = getattr(model, "backbone", None)
    network = getattr(backbone, "network", None)
    if network is None:
        raise RuntimeError("Expected custom TransXNet backbone with a 'network' attribute.")
    if layer_key == "stage3_h16" and len(network) >= 5 and hasattr(network[4], "__len__") and len(network[4]) > 0:
        return network[4][-1]
    if layer_key == "stage4_h32" and len(network) >= 7 and hasattr(network[6], "__len__") and len(network[6]) > 0:
        return network[6][-1]
    for stage in reversed(list(network)):
        if hasattr(stage, "__len__") and len(stage) > 0:
            return stage[-1]
    raise RuntimeError("Could not find a spatial target layer for Grad-CAM.")


def make_dataset(module, case: GradCamCase):
    dataset_cls = getattr(module, case.dataset_class)
    root_dir = getattr(module.Config, case.root_attr)
    return dataset_cls(
        root_dir,
        case.split,
        transform=None,
        use_roi_crop=True,
        bbox_expand_ratio=case.bbox_expand,
        min_crop_size=module.Config.min_crop_size,
        use_whole_image_fallback=module.Config.use_whole_image_fallback,
    )


def get_roi_by_id(dataset, image_id: str) -> tuple[Image.Image, int, str]:
    for idx, sample in enumerate(dataset.samples):
        if str(sample["image_id"]) == image_id:
            img, label, got_id, _path = dataset[idx]
            return img, int(label), str(got_id)
    raise KeyError(f"{image_id} not found in {dataset.split}")


def resize_for_display(img: Image.Image, size: int = 256) -> np.ndarray:
    return np.asarray(img.resize((size, size), Image.BICUBIC).convert("RGB"), dtype=np.float32) / 255.0


def cam_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heat = plt.get_cmap("magma")(cam)[..., :3]
    return np.clip((1.0 - alpha) * rgb + alpha * heat, 0.0, 1.0)


def center_energy(cam: np.ndarray, frac: float = 0.68) -> float:
    h, w = cam.shape
    y0 = int(round((1.0 - frac) * 0.5 * h))
    y1 = int(round((1.0 + frac) * 0.5 * h))
    x0 = int(round((1.0 - frac) * 0.5 * w))
    x1 = int(round((1.0 + frac) * 0.5 * w))
    total = float(cam.sum()) + 1e-8
    return float(cam[y0:y1, x0:x1].sum()) / total


def build_panel(case: GradCamCase, device: torch.device) -> dict[str, Any]:
    module = import_train_module(case.module_name)
    dataset = make_dataset(module, case)
    _train_transform, eval_transform = module.build_transforms()

    model = instantiate_model(module, case.run_dir / case.checkpoint_name, device)
    target_layer = get_target_layer(model, case.target_layer_key)
    gradcam = GradCAM(model, target_layer)
    try:
        best: dict[str, Any] | None = None
        for prediction in select_candidates(case):
            roi_img, label, image_id = get_roi_by_id(dataset, prediction["image_id"])
            tensor = eval_transform(roi_img).unsqueeze(0).to(device)
            target_class = int(label)
            cam, probs = gradcam(tensor, target_class)
            rgb = resize_for_display(roi_img, size=256)
            conf_true = float(probs[target_class])
            centrality = center_energy(cam)
            white_fraction = float((rgb.mean(axis=2) > 0.90).mean())
            score = 0.62 * centrality + 0.28 * conf_true + 0.10 * float(prediction["prob_class1"]) - 0.80 * white_fraction
            item = {
                "dataset": case.dataset,
                "image_id": image_id,
                "true_label": int(label),
                "pred_label_csv": int(prediction["pred_label"]),
                "prob_class1_csv": float(prediction["prob_class1"]),
                "prob_true_single_ckpt": conf_true,
                "threshold": float(prediction["threshold"]),
                "target_layer": type(target_layer).__name__,
                "target_layer_key": case.target_layer_key,
                "activation_size": f"{cam.shape[0]}x{cam.shape[1]} displayed; target={case.target_layer_key}",
                "center_energy": centrality,
                "white_fraction": white_fraction,
                "score": score,
                "rgb": rgb,
                "overlay": cam_overlay(rgb, cam),
                "cam": cam,
            }
            if best is None or item["score"] > best["score"]:
                best = item
        if best is None:
            raise RuntimeError(f"No Grad-CAM panel could be built for {case.dataset}")
        return best
    finally:
        gradcam.close()


def save_figure(panels: list[dict[str, Any]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(panels),
        figsize=(6.2, 2.95),
        constrained_layout=False,
    )

    for col, panel in enumerate(panels):
        axes[0, col].set_title(panel["dataset"], fontsize=8.8, fontweight="bold", pad=4)
        for row, image in enumerate([panel["rgb"], panel["overlay"]]):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
                spine.set_color("#222222")

    axes[0, 0].set_ylabel("ROI input", rotation=90, ha="center", va="center", labelpad=12, fontsize=8.2)
    axes[1, 0].set_ylabel("Grad-CAM", rotation=90, ha="center", va="center", labelpad=12, fontsize=8.2)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.035, wspace=0.045, hspace=0.08)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(OUT_BASE.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panels = [build_panel(case, device) for case in CASES]
    save_figure(panels)
    selected_path = OUT_BASE.with_name(OUT_BASE.name + "_selected_cases.json")
    serializable = [
        {
            k: v
            for k, v in panel.items()
            if k not in {"rgb", "overlay", "cam"}
        }
        for panel in panels
    ]
    selected_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] saved {OUT_BASE.with_suffix('.png')}")
    print(f"[OK] saved {OUT_BASE.with_suffix('.pdf')}")
    print(f"[OK] saved {OUT_BASE.with_suffix('.svg')}")
    print(f"[OK] selected cases -> {selected_path}")


if __name__ == "__main__":
    main()
