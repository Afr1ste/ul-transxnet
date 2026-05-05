from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIG: Dict[str, object] = {
    # 若为 None，会自动向上查找名为 Thyroid 的项目根目录
    "PROJECT_ROOT": None,
    "PROJECT_ROOT_NAME": "Thyroid",

    # 当前主线架构（与 fl_tn5000_roi_train_refined_A_seed17.py / E06 主线一致）
    "BACKBONE_MODULE": "models.transxnetggg",
    "BACKBONE_FUNC": "transxnet_t",
    "BACKBONE_OUT_DIM": 1000,
    "HEAD_HIDDEN_DIM": 512,
    "HEAD_DROPOUT": 0.30,
    "INPUT_SIZE": 256,
    "NUM_CLASSES": 2,
    "OPSET_VERSION": 17,

    # ONNX 输出位置
    "ONNX_DIR": r"Android\models\onnx",
    "APP_ASSETS_DIR": r"Android\app\src\main\assets",
    "COPY_TO_APP_ASSETS": True,

    # 导出日志位置
    "EXPORT_LOG_DIR": r"Android\export\export_logs",
}


EXPORT_JOBS: List[Dict[str, str]] = [
    {
        # Mobile-Fast / Accurate 默认主模型（epoch062 的别名）
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch062_bal_acc_0.9208.pth",
        "onnx_name": "tn5000_current_mainline.onnx",
        "log_name": "export_tn5000_current_mainline.log",
    },
    {
        # Mobile-Ensemble 第 2 个模型
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch060_bal_acc_0.9195.pth",
        "onnx_name": "tn5000_epoch060_bal_acc_0.9195.onnx",
        "log_name": "export_tn5000_epoch060.log",
    },
    {
        # Mobile-Ensemble 第 3 个模型
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch054_bal_acc_0.9169.pth",
        "onnx_name": "tn5000_epoch054_bal_acc_0.9169.onnx",
        "log_name": "export_tn5000_epoch054.log",
    },
]


def find_project_root() -> Path:
    cfg_root = CONFIG.get("PROJECT_ROOT")
    if cfg_root:
        root = Path(str(cfg_root)).resolve()
        if not root.exists():
            raise FileNotFoundError(f"PROJECT_ROOT 不存在: {root}")
        return root

    cur = Path(__file__).resolve()
    root_name = str(CONFIG["PROJECT_ROOT_NAME"])
    for p in [cur.parent] + list(cur.parents):
        if p.name == root_name:
            return p
    raise RuntimeError(
        f"无法自动定位项目根目录 {root_name}，请在 CONFIG['PROJECT_ROOT'] 中手动填写 Thyroid 根目录。"
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


ONNX_DIR = resolve_path(str(CONFIG["ONNX_DIR"]))
APP_ASSETS_DIR = resolve_path(str(CONFIG["APP_ASSETS_DIR"]))
EXPORT_LOG_DIR = resolve_path(str(CONFIG["EXPORT_LOG_DIR"]))


@contextlib.contextmanager

def silence_output():
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield buf_out, buf_err


def load_backbone_fn(module_name: str, func_name: str):
    import importlib

    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise AttributeError(f"{module_name} 中不存在 {func_name}")
    return getattr(module, func_name)


class HighResMambaExport(nn.Module):
    """严格对齐当前 TN5000 主线网络本体。"""

    def __init__(self):
        super().__init__()
        backbone_fn = load_backbone_fn(str(CONFIG["BACKBONE_MODULE"]), str(CONFIG["BACKBONE_FUNC"]))
        self.backbone = backbone_fn(
            num_classes=int(CONFIG["BACKBONE_OUT_DIM"]),
            img_size=int(CONFIG["INPUT_SIZE"]),
        )
        self.head = nn.Sequential(
            nn.Linear(int(CONFIG["BACKBONE_OUT_DIM"]), int(CONFIG["HEAD_HIDDEN_DIM"])),
            nn.GELU(),
            nn.Dropout(float(CONFIG["HEAD_DROPOUT"])),
            nn.Linear(int(CONFIG["HEAD_HIDDEN_DIM"]), int(CONFIG["NUM_CLASSES"])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


@torch.no_grad()
def export_safe_resize_pool(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    导出专用替代池化：
    - 若输入空间尺寸可整除目标尺寸，则用 avg_pool2d
    - 否则退化为 bilinear resize，绕开部分旧导出器对 adaptive_avg_pool2d 的兼容问题
    """
    h = int(x.shape[-2])
    w = int(x.shape[-1])

    if h == out_h and w == out_w:
        return x

    if h % out_h == 0 and w % out_w == 0:
        kh = h // out_h
        kw = w // out_w
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))

    return F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)


def apply_dynamicconv_export_patch() -> None:
    import importlib

    mod = importlib.import_module(str(CONFIG["BACKBONE_MODULE"]))

    if getattr(mod.DynamicConv2d, "_oai_export_patch_applied", False):
        return

    original_forward = mod.DynamicConv2d.forward

    def patched_forward(self, x):
        b, c, h, w = x.shape

        pooled = export_safe_resize_pool(x, self.K, self.K)
        scale = self.proj(pooled).reshape(b, self.num_groups, c, self.K, self.K)
        scale = torch.softmax(scale, dim=1)

        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            bias_scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            bias_scale = torch.softmax(bias_scale.reshape(b, self.num_groups, c), dim=1)
            bias = bias_scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
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

    mod.DynamicConv2d._oai_original_forward = original_forward
    mod.DynamicConv2d.forward = patched_forward
    mod.DynamicConv2d._oai_export_patch_applied = True


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def try_extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "ema_state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return normalize_state_dict_keys(ckpt_obj[key])
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return normalize_state_dict_keys(ckpt_obj)
    raise RuntimeError("无法从 checkpoint 中解析 state_dict。")


def load_checkpoint_flex(model: nn.Module, ckpt_path: Path) -> nn.Module:
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = try_extract_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("=" * 80)
    print("load_state_dict result")
    print("=" * 80)
    print(f"missing keys    : {len(missing)}")
    if missing:
        print("  -> examples:", missing[:20])
    print(f"unexpected keys : {len(unexpected)}")
    if unexpected:
        print("  -> examples:", unexpected[:20])
    return model


def export_onnx(model: nn.Module, onnx_path: Path, export_log_path: Path) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, 3, int(CONFIG["INPUT_SIZE"]), int(CONFIG["INPUT_SIZE"]), dtype=torch.float32)

    with torch.no_grad():
        y = model(dummy)
    print(f"PyTorch warmup output shape: {tuple(y.shape)}")

    export_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write("[INFO] export started\n")
        try:
            print("[INFO] exporting with torch.onnx.export(..., dynamo=False)")
            with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                torch.onnx.export(
                    model,
                    args=(dummy,),
                    f=str(onnx_path),
                    input_names=["image"],
                    output_names=["logits"],
                    export_params=True,
                    opset_version=int(CONFIG["OPSET_VERSION"]),
                    do_constant_folding=True,
                    dynamo=False,
                )
            print(f"[OK] ONNX saved to: {onnx_path}")
        except Exception:
            traceback.print_exc(file=logf)
            raise


def maybe_copy_to_assets(onnx_path: Path) -> None:
    if not bool(CONFIG["COPY_TO_APP_ASSETS"]):
        return
    APP_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    dst = APP_ASSETS_DIR / onnx_path.name
    dst.write_bytes(onnx_path.read_bytes())
    print(f"[COPY] copied to app assets: {dst}")


def export_one(job: Dict[str, str]) -> None:
    ckpt_path = resolve_path(job["ckpt_path"])
    onnx_path = ONNX_DIR / job["onnx_name"]
    export_log_path = EXPORT_LOG_DIR / job["log_name"]

    print("\n" + "=" * 80)
    print(f"Export: {job['onnx_name']}")
    print("=" * 80)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"CKPT_PATH    : {ckpt_path}")
    print(f"ONNX_PATH    : {onnx_path}")
    print(f"INPUT_SIZE   : {CONFIG['INPUT_SIZE']}")
    print(f"NUM_CLASSES  : {CONFIG['NUM_CLASSES']}")
    print(f"EXPORT_LOG   : {export_log_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    apply_dynamicconv_export_patch()
    model = HighResMambaExport()
    model = load_checkpoint_flex(model, ckpt_path)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    export_onnx(model, onnx_path, export_log_path)
    maybe_copy_to_assets(onnx_path)


def main() -> None:
    print("=" * 80)
    print("Export TN5000 ONNX Suite")
    print("=" * 80)
    print(f"PROJECT_ROOT       : {PROJECT_ROOT}")
    print(f"ONNX_DIR           : {ONNX_DIR}")
    print(f"APP_ASSETS_DIR     : {APP_ASSETS_DIR}")
    print(f"COPY_TO_APP_ASSETS : {CONFIG['COPY_TO_APP_ASSETS']}")
    print(f"NUM_JOBS           : {len(EXPORT_JOBS)}")

    for job in EXPORT_JOBS:
        export_one(job)

    print("\n[DONE] All ONNX exports finished.")


if __name__ == "__main__":
    main()
