from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


CONFIG: Dict[str, object] = {
    # 若为 None，会自动向上查找名为 Thyroid 的项目根目录
    "PROJECT_ROOT": None,
    "PROJECT_ROOT_NAME": "Thyroid",

    # 架构与导出脚本保持一致
    "BACKBONE_MODULE": "models.transxnetggg",
    "BACKBONE_FUNC": "transxnet_t",
    "BACKBONE_OUT_DIM": 1000,
    "HEAD_HIDDEN_DIM": 512,
    "HEAD_DROPOUT": 0.30,
    "INPUT_SIZE": 256,
    "NUM_CLASSES": 2,

    # True = 用同 checkpoint 的 PyTorch 模型与 ORT 做随机输入数值对齐
    "DO_PYTORCH_COMPARE": True,
}


CHECK_JOBS: List[Dict[str, str]] = [
    {
        "onnx_path": r"Android\models\onnx\tn5000_current_mainline.onnx",
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch062_bal_acc_0.9208.pth",
    },
    {
        "onnx_path": r"Android\models\onnx\tn5000_epoch060_bal_acc_0.9195.onnx",
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch060_bal_acc_0.9195.pth",
    },
    {
        "onnx_path": r"Android\models\onnx\tn5000_epoch054_bal_acc_0.9169.onnx",
        "ckpt_path": r"tn5000_roi_runs_E06_mainline_multiseed\20260329_024552\epoch054_bal_acc_0.9169.pth",
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


def load_backbone_fn(module_name: str, func_name: str):
    import importlib

    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise AttributeError(f"{module_name} 中不存在 {func_name}")
    return getattr(module, func_name)


class HighResMambaExport(nn.Module):
    def __init__(
        self,
        backbone_module: str,
        backbone_func: str,
        backbone_out_dim: int,
        hidden_dim: int,
        num_classes: int,
        input_size: int,
        dropout: float,
    ):
        super().__init__()
        backbone_fn = load_backbone_fn(backbone_module, backbone_func)
        self.backbone = backbone_fn(num_classes=backbone_out_dim, img_size=input_size)
        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def build_model(cfg: Dict[str, object]) -> nn.Module:
    return HighResMambaExport(
        backbone_module=str(cfg["BACKBONE_MODULE"]),
        backbone_func=str(cfg["BACKBONE_FUNC"]),
        backbone_out_dim=int(cfg["BACKBONE_OUT_DIM"]),
        hidden_dim=int(cfg["HEAD_HIDDEN_DIM"]),
        num_classes=int(cfg["NUM_CLASSES"]),
        input_size=int(cfg["INPUT_SIZE"]),
        dropout=float(cfg["HEAD_DROPOUT"]),
    )


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


def load_checkpoint_flex(model: nn.Module, ckpt_path: Path) -> None:
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = try_extract_state_dict(ckpt_obj)
    model.load_state_dict(state_dict, strict=False)


def check_one(job: Dict[str, str]) -> None:
    cfg = dict(CONFIG)
    onnx_path = resolve_path(job["onnx_path"])
    ckpt_path = resolve_path(job["ckpt_path"])

    if not onnx_path.exists():
        raise FileNotFoundError(f"找不到 ONNX 文件: {onnx_path}")

    print("\n" + "=" * 80)
    print(f"Check: {onnx_path.name}")
    print("=" * 80)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"ONNX_PATH    : {onnx_path}")
    print(f"CKPT_PATH    : {ckpt_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[OK] onnx.checker.check_model passed")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ORT inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ORT outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, int(cfg["INPUT_SIZE"]), int(cfg["INPUT_SIZE"])).astype(np.float32)
    ort_out = sess.run(None, {input_name: x})[0]
    print("[OK] ORT forward passed")
    print("ORT output shape:", ort_out.shape)
    print("ORT output sample:", ort_out[0])

    if bool(cfg["DO_PYTORCH_COMPARE"]):
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DO_PYTORCH_COMPARE=True，但找不到 checkpoint: {ckpt_path}")

        model = build_model(cfg).eval().cpu()
        load_checkpoint_flex(model, ckpt_path)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x)).cpu().numpy()

        abs_diff = np.abs(pt_out - ort_out)
        print("[COMPARE] max_abs_diff :", float(abs_diff.max()))
        print("[COMPARE] mean_abs_diff:", float(abs_diff.mean()))

        if float(abs_diff.max()) < 1e-3:
            print("[GOOD] PyTorch vs ORT 数值对齐很好。")
        elif float(abs_diff.max()) < 1e-2:
            print("[OK] 数值差异可接受，后续再用真实图片继续确认。")
        else:
            print("[WARN] 数值差异偏大，建议检查：")
            print("       1) checkpoint 是否和导出用的一致")
            print("       2) 是否误改了 backbone/head 结构")
            print("       3) 导出路径是否走了不兼容算子")


def main() -> None:
    print("=" * 80)
    print("Check TN5000 ONNX Suite")
    print("=" * 80)
    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    print(f"DO_PYTORCH_COMPARE  : {CONFIG['DO_PYTORCH_COMPARE']}")
    print(f"NUM_JOBS            : {len(CHECK_JOBS)}")

    for job in CHECK_JOBS:
        check_one(job)

    print("\n[DONE] All ONNX checks finished.")


if __name__ == "__main__":
    main()
