"""Run a lightweight forward-pass smoke test for published model code.

This script intentionally uses random input and does not require any medical
image dataset or trained checkpoint. It is meant to verify that the artifact can
construct a model and execute a forward pass in the declared environment.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UL-TransXNet artifact smoke test")
    p.add_argument("--model", default="ul-transxnet", choices=["ul-transxnet", "transxnet-gg", "transxnet-g", "transxnet"])
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--verbose", action="store_true", help="Do not suppress constructor debug prints")
    return p.parse_args()


def load_factory(model_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    if model_name == "ul-transxnet":
        from models.transxnetggg import transxnet_t
        return transxnet_t
    if model_name == "transxnet-gg":
        from models.transxnetgg import transxnet_t
        return transxnet_t
    if model_name == "transxnet-g":
        from models.transxnetg import transxnet_t
        return transxnet_t
    if model_name == "transxnet":
        from models.transxnet import transxnet_t
        return transxnet_t
    raise ValueError(model_name)


def main() -> int:
    args = parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency diagnostic path
        print(json.dumps({"ok": False, "error": f"PyTorch import failed: {exc}"}, indent=2))
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "CUDA requested but torch.cuda.is_available() is false"}, indent=2))
        return 2

    try:
        factory = load_factory(args.model)
        sink = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            model = factory(num_classes=args.num_classes)
        model.eval().to(args.device)

        x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=args.device)
        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        n_params = sum(p.numel() for p in model.parameters())
        result = {
            "ok": True,
            "model": args.model,
            "device": args.device,
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
            "parameters": n_params,
            "elapsed_ms": round(elapsed_ms, 3),
        }
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover - diagnostic path
        result = {"ok": False, "model": args.model, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(result, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
