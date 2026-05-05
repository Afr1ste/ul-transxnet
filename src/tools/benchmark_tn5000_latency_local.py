from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_tn5000_roi_compare_multimodel import TN5000ROIDataset  # noqa: E402
from tools.compute_paper_complexity_table import MODEL_SPECS, ModelSpec, PaperClassifier, profile_spec  # noqa: E402


OUT_DIR = PROJECT_ROOT / "eval_reports" / "tn5000_local_latency"
DEFAULT_TN5000_ROOT = PROJECT_ROOT / "TN5000_forReview"
BENCHMARK_METHODS = [
    "MobileNetV3-Large",
    "EfficientNet-B0",
    "EfficientFormer-L1",
    "RepViT-M1.1",
    "TransXNet-GGG-MCA",
    "DenseNet121",
    "ResNet50",
    "ConvNeXt-Tiny",
    "Swin-T",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def make_loader(tn5000_root: Path, spec: ModelSpec, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((spec.input_size, spec.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = TN5000ROIDataset(
        root_dir=tn5000_root,
        split="test",
        transform=transform,
        use_roi_crop=True,
        bbox_expand_ratio=0.30,
        min_crop_size=64,
        use_whole_image_fallback=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def benchmark_model_only(model: torch.nn.Module, device: torch.device, input_size: int, warmup: int, repeats: int) -> dict:
    x = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(starter.elapsed_time(ender)))
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    return summarize_times(times_ms)


@torch.no_grad()
def benchmark_tn5000_pipeline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None,
) -> dict:
    model.eval()
    times_ms: list[float] = []
    seen = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for images, _labels, _ids, _paths in loader:
        if max_samples is not None and seen >= max_samples:
            break
        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        seen += int(images.shape[0])
    out = summarize_times(times_ms)
    out["samples"] = seen
    return out


def summarize_times(times_ms: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(times_ms) if times_ms else float("nan"),
        "median_ms": statistics.median(times_ms) if times_ms else float("nan"),
        "p95_ms": percentile(times_ms, 0.95),
        "fps": 1000.0 / statistics.fmean(times_ms) if times_ms and statistics.fmean(times_ms) > 0 else float("nan"),
        "n": len(times_ms),
    }


def fmt_float(v: object, digits: int = 4) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.{digits}f}"
    return str(v)


def format_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            if key in {"params_m", "macs_g"}:
                out[key] = f"{value:.2f}"
            else:
                out[key] = f"{value:.4f}"
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Local latency benchmark on TN5000 test ROI inputs.")
    parser.add_argument("--tn5000-root", default=str(DEFAULT_TN5000_ROOT))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--methods", default=",".join(BENCHMARK_METHODS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "latest.status.json"
    csv_path = out_dir / f"tn5000_local_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    latest_csv = out_dir / "tn5000_local_latency_latest.csv"

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    specs_by_name = {s.method: s for s in MODEL_SPECS}
    specs = [specs_by_name[m] for m in requested]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")

    status = {
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "device": str(device),
        "tn5000_root": str(Path(args.tn5000_root)),
        "csv_path": str(csv_path),
        "latest_csv": str(latest_csv),
        "methods_total": len(specs),
        "methods_done": 0,
        "current_method": None,
        "last_error": None,
    }
    write_json(status_path, status)

    rows: list[dict] = []
    for index, spec in enumerate(specs, start=1):
        status.update({"updated_at": now_iso(), "current_method": spec.method, "methods_done": index - 1})
        write_json(status_path, status)
        print(f"[{index}/{len(specs)}] {spec.method} input={spec.input_size} device={device}", flush=True)

        try:
            complexity = profile_spec(spec)
            model = PaperClassifier(spec).to(device).eval()
            model_only = benchmark_model_only(
                model=model,
                device=device,
                input_size=spec.input_size,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            loader = make_loader(Path(args.tn5000_root), spec, num_workers=args.num_workers)
            pipeline = benchmark_tn5000_pipeline(
                model=model,
                loader=loader,
                device=device,
                max_samples=args.max_samples if args.max_samples > 0 else None,
            )
            row = {
                "method": spec.method,
                "device": str(device),
                "input": f"{spec.input_size}x{spec.input_size}",
                "params_m": float(complexity["params_m"]),
                "macs_g": float(complexity["macs_g"]),
                "model_mean_ms": float(model_only["mean_ms"]),
                "model_median_ms": float(model_only["median_ms"]),
                "model_p95_ms": float(model_only["p95_ms"]),
                "model_fps": float(model_only["fps"]),
                "model_repeats": int(model_only["n"]),
                "pipeline_mean_ms": float(pipeline["mean_ms"]),
                "pipeline_median_ms": float(pipeline["median_ms"]),
                "pipeline_p95_ms": float(pipeline["p95_ms"]),
                "pipeline_fps": float(pipeline["fps"]),
                "pipeline_samples": int(pipeline["samples"]),
            }
            rows.append(row)
            append_csv(csv_path, format_row(row))
            latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(
                "[DONE] {method} model_mean={model_mean_ms:.3f}ms p95={model_p95_ms:.3f}ms "
                "pipeline_mean={pipeline_mean_ms:.3f}ms p95={pipeline_p95_ms:.3f}ms".format(**row),
                flush=True,
            )
        except Exception as exc:
            status.update(
                {
                    "state": "failed",
                    "updated_at": now_iso(),
                    "current_method": spec.method,
                    "last_error": repr(exc),
                }
            )
            write_json(status_path, status)
            raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    status.update(
        {
            "state": "completed",
            "updated_at": now_iso(),
            "current_method": None,
            "methods_done": len(specs),
            "rows": len(rows),
        }
    )
    write_json(status_path, status)
    print(f"[SAVE] {csv_path}", flush=True)
    print(f"[SAVE] {latest_csv}", flush=True)
    print(f"[SAVE] {status_path}", flush=True)


if __name__ == "__main__":
    main()
