from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Overwriting .* in registry.*")


@dataclass(frozen=True)
class ModelSpec:
    method: str
    role: str
    model_family: str
    input_size: int
    backbone_name: str = ""
    backbone_module: str = ""
    backbone_func: str = ""
    backbone_out_dim: int = 0


MODEL_SPECS: list[ModelSpec] = [
    # Finalized benchmark pool.
    ModelSpec("ResNet50", "Prior baseline", "timm", 256, backbone_name="resnet50"),
    ModelSpec("DenseNet121", "Prior baseline", "timm", 256, backbone_name="densenet121"),
    ModelSpec("EfficientNet-B0", "Prior baseline", "timm", 256, backbone_name="efficientnet_b0"),
    ModelSpec("MobileNetV3-Large", "Prior baseline", "timm", 256, backbone_name="mobilenetv3_large_100"),
    ModelSpec("Swin-T", "Prior baseline", "timm", 256, backbone_name="swin_tiny_patch4_window7_224"),
    ModelSpec("ConvNeXt-Tiny", "Prior baseline", "timm", 256, backbone_name="convnext_tiny"),
    ModelSpec("RepViT-M1.1", "Prior baseline", "timm", 256, backbone_name="repvit_m1_1"),
    # EfficientFormer-L1 is trained at 224 in the comparison scripts.
    ModelSpec("EfficientFormer-L1", "Prior baseline", "timm", 224, backbone_name="efficientformer_l1"),
    # Progressive TransXNet-family variants used in the current TN5000 ablation.
    ModelSpec(
        "TransXNet",
        "Ours / ablation",
        "custom",
        256,
        backbone_name="transxnet_t",
        backbone_module="models.transxnet",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
    ),
    ModelSpec(
        "TransXNet-G",
        "Ours / ablation",
        "custom",
        256,
        backbone_name="transxnet_t",
        backbone_module="models.transxnetg",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
    ),
    ModelSpec(
        "TransXNet-GG",
        "Ours / ablation",
        "custom",
        256,
        backbone_name="transxnet_t",
        backbone_module="models.transxnetgg",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
    ),
    ModelSpec(
        "TransXNet-GGG-noMCA",
        "Ours / ablation",
        "custom",
        256,
        backbone_name="transxnet_t",
        backbone_module="models.transxnetggg_nomca",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
    ),
    ModelSpec(
        "TransXNet-GGG-MCA",
        "Ours",
        "custom",
        256,
        backbone_name="transxnet_t",
        backbone_module="models.transxnetggg",
        backbone_func="transxnet_t",
        backbone_out_dim=1000,
    ),
]


class PaperClassifier(nn.Module):
    """Classifier wrapper matching the training scripts, but without downloads."""

    def __init__(self, spec: ModelSpec, num_classes: int = 2, dropout: float = 0.30):
        super().__init__()
        self.spec = spec
        if spec.model_family == "timm":
            import timm

            create_kwargs = dict(pretrained=False, num_classes=0, global_pool="avg")
            if any(k in spec.backbone_name.lower() for k in ("swin", "vit", "deit", "beit")):
                create_kwargs["img_size"] = spec.input_size
            self.backbone = timm.create_model(spec.backbone_name, **create_kwargs)
            feat_dim = self._infer_feat_dim(spec.input_size)
        elif spec.model_family == "custom":
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                module = importlib.import_module(spec.backbone_module)
            backbone_fn = getattr(module, spec.backbone_func)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.backbone = backbone_fn(
                    pretrained=False,
                    num_classes=spec.backbone_out_dim,
                    img_size=spec.input_size,
                )
            feat_dim = int(spec.backbone_out_dim)
        else:
            raise ValueError(f"Unsupported model_family: {spec.model_family}")

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @torch.no_grad()
    def _infer_feat_dim(self, input_size: int) -> int:
        self.eval()
        dummy = torch.zeros(1, 3, input_size, input_size)
        feats = self.extract_features(dummy)
        if feats.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape for {self.spec.method}: {tuple(feats.shape)}")
        return int(feats.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Some custom model constructors print diagnostic text; keep profiling output clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        elif feats.ndim == 3:
            feats = feats.mean(dim=1)
        elif feats.ndim == 1:
            feats = feats.unsqueeze(0)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_spec(spec: ModelSpec) -> dict[str, object]:
    torch.set_num_threads(1)
    model = PaperClassifier(spec).eval()
    dummy = torch.zeros(1, 3, spec.input_size, spec.input_size)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    params = count_trainable_params(model)
    return {
        "method": spec.method,
        "role": spec.role,
        "input": f"{spec.input_size}x{spec.input_size}",
        "params_m": params / 1e6,
        "macs_g": macs / 1e9,
    }


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def latex_input_size(text: str) -> str:
    if "x" in text:
        left, right = text.split("x", 1)
        if left.isdigit() and right.isdigit():
            return f"${left} \\times {right}$"
    return latex_escape(text)


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paper_model_complexity_table.csv"
    tex_path = out_dir / "paper_model_complexity_table.tex"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "role", "input", "params_m", "macs_g"])
        writer.writeheader()
        for row in rows:
            row_for_csv = dict(row)
            row_for_csv["params_m"] = f"{float(row['params_m']):.2f}"
            row_for_csv["macs_g"] = f"{float(row['macs_g']):.2f}"
            writer.writerow(row_for_csv)

    lines = [
        "% Auto-generated by tools/compute_paper_complexity_table.py",
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\small",
        "    \\caption{Model complexity of the finalized comparison pool and TransXNet-family variants. Parameters and MACs are measured for the full binary classifier, including the lightweight MLP head. MACs are reported for one forward pass at the listed input size.}",
        "    \\label{tab:complexity}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{llccc}",
        "        \\toprule",
        "        Method & Role & Input & Params (M) & MACs (G) \\\\",
        "        \\midrule",
    ]
    for row in rows:
        method = latex_escape(str(row["method"]))
        role = latex_escape(str(row["role"]))
        input_size = latex_input_size(str(row["input"]))
        params_m = f"{float(row['params_m']):.2f}"
        macs_g = f"{float(row['macs_g']):.2f}"
        if row["method"] == "TransXNet-GGG-MCA":
            method = f"\\textbf{{{method}}}"
            role = f"\\textbf{{{role}}}"
            input_size = f"\\textbf{{{input_size}}}"
            params_m = f"\\textbf{{{params_m}}}"
            macs_g = f"\\textbf{{{macs_g}}}"
        lines.append(f"        {method} & {role} & {input_size} & {params_m} & {macs_g} \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "\\end{table*}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {csv_path}")
    print(f"[SAVE] {tex_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paper complexity table for trained comparison models.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        print(f"[PROFILE] {spec.method} ({spec.input_size}x{spec.input_size})")
        row = profile_spec(spec)
        rows.append(row)
        print(f"  params={row['params_m']:.2f}M macs={row['macs_g']:.2f}G")
    write_outputs(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
