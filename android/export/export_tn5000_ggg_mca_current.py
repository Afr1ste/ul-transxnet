from __future__ import annotations

import contextlib
import importlib
import io
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BACKBONE_MODULE = "models.transxnetggg"
BACKBONE_FUNC = "transxnet_t"
INPUT_SIZE = 256
BACKBONE_OUT_DIM = 1000
HEAD_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.30
NUM_CLASSES = 2
OPSET_VERSION = 17

RUN_DIR = PROJECT_ROOT / "tn5000_roi_runs_ggg_mca_enabled_3seed" / "20260426_121126"
EXPORT_JOBS = [
    (
        RUN_DIR / "best_model_tn5000_roi.pth",
        "tn5000_ggg_mca_s27_current.onnx",
        "export_tn5000_ggg_mca_s27_current.log",
    ),
    (
        RUN_DIR / "epoch047_auc_0.9619.pth",
        "tn5000_ggg_mca_s27_epoch047.onnx",
        "export_tn5000_ggg_mca_s27_epoch047.log",
    ),
    (
        RUN_DIR / "epoch046_auc_0.9613.pth",
        "tn5000_ggg_mca_s27_epoch046.onnx",
        "export_tn5000_ggg_mca_s27_epoch046.log",
    ),
]
ONNX_DIR = PROJECT_ROOT / "Android" / "models" / "onnx"
EXPORT_LOG_DIR = PROJECT_ROOT / "Android" / "export" / "export_logs"
ASSET_DIRS = [
    PROJECT_ROOT / "Android" / "app" / "src" / "main" / "assets",
    PROJECT_ROOT / "TN5000OrtDemoComplete" / "app" / "src" / "main" / "assets",
]


def load_backbone_fn():
    module = importlib.import_module(BACKBONE_MODULE)
    return getattr(module, BACKBONE_FUNC)


class Tn5000GggMcaExport(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone_fn = load_backbone_fn()
        self.backbone = backbone_fn(
            num_classes=BACKBONE_OUT_DIM,
            img_size=INPUT_SIZE,
        )
        self.head = nn.Sequential(
            nn.Linear(BACKBONE_OUT_DIM, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@torch.no_grad()
def export_safe_resize_pool(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    h = int(x.shape[-2])
    w = int(x.shape[-1])
    if h == out_h and w == out_w:
        return x
    if h % out_h == 0 and w % out_w == 0:
        return F.avg_pool2d(x, kernel_size=(h // out_h, w // out_w), stride=(h // out_h, w // out_w))
    return F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)


def apply_dynamicconv_export_patch() -> None:
    mod = importlib.import_module(BACKBONE_MODULE)
    if getattr(mod.DynamicConv2d, "_ggg_mca_export_patch_applied", False):
        return

    original_forward = mod.DynamicConv2d.forward

    def patched_forward(self, x):
        b, c, h, w = x.shape
        pooled = export_safe_resize_pool(x, self.K, self.K)
        scale = self.proj(pooled).reshape(b, self.num_groups, c, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = (scale * self.weight.unsqueeze(0)).sum(dim=1)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            bias_scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            bias_scale = torch.softmax(bias_scale.reshape(b, self.num_groups, c), dim=1)
            bias = (bias_scale * self.bias.unsqueeze(0)).sum(dim=1).flatten(0)
        else:
            bias = None

        out = F.conv2d(
            x.reshape(1, -1, h, w),
            weight=weight,
            padding=self.K // 2,
            groups=b * c,
            bias=bias,
        )
        return out.reshape(b, c, h, w)

    mod.DynamicConv2d._ggg_mca_original_forward = original_forward
    mod.DynamicConv2d.forward = patched_forward
    mod.DynamicConv2d._ggg_mca_export_patch_applied = True


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    state_dict = normalize_state_dict_keys(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys={len(missing)}")
    if missing:
        print("missing_examples=", missing[:20])
    print(f"unexpected_keys={len(unexpected)}")
    if unexpected:
        print("unexpected_examples=", unexpected[:20])


def maybe_check_onnx(onnx_path: Path) -> None:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("[OK] onnx.checker passed")
    except Exception as exc:
        print(f"[WARN] onnx.checker skipped/failed: {exc}")


def export_one(checkpoint: Path, onnx_name: str, log_name: str) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / onnx_name
    log_path = EXPORT_LOG_DIR / log_name

    print(f"project_root={PROJECT_ROOT}")
    print(f"checkpoint={checkpoint}")
    print(f"onnx_path={onnx_path}")
    print(f"asset_dirs={[str(p) for p in ASSET_DIRS]}")

    apply_dynamicconv_export_patch()
    model = Tn5000GggMcaExport().eval().cpu()
    load_checkpoint(model, checkpoint)

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy)
    print(f"pytorch_output_shape={tuple(output.shape)}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write("[INFO] export started\n")
        try:
            with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                torch.onnx.export(
                    model,
                    args=(dummy,),
                    f=str(onnx_path),
                    input_names=["image"],
                    output_names=["logits"],
                    export_params=True,
                    opset_version=OPSET_VERSION,
                    do_constant_folding=True,
                    dynamo=False,
                )
        except Exception:
            traceback.print_exc(file=logf)
            raise

    maybe_check_onnx(onnx_path)
    for asset_dir in ASSET_DIRS:
        asset_dir.mkdir(parents=True, exist_ok=True)
        dst = asset_dir / onnx_name
        dst.write_bytes(onnx_path.read_bytes())
        print(f"[COPY] {dst}")

    print("[DONE] export complete")


def export() -> None:
    for checkpoint, onnx_name, log_name in EXPORT_JOBS:
        print("=" * 80)
        print(f"Export {onnx_name}")
        print("=" * 80)
        export_one(checkpoint, onnx_name, log_name)


if __name__ == "__main__":
    export()
