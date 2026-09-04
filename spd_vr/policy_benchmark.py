"""Benchmark the real SPD policy's streaming inference path.

This command intentionally requires a user-supplied, compatible DINOv3
checkpoint. It never downloads weights, silently falls back to a tiny model,
or treats a CPU result as a 30 Hz deployment qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from .config import SPDModelConfig
from .contracts import CAMERA_NAMES, HISTORY_STEPS, IMAGE_STRIDE, ROBOT_DOF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def benchmark_policy(
    dino_checkpoint: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 1,
    warmup_ticks: int = 2,
    measure_ticks: int = 16,
    compile_model: bool = False,
    euler_steps: int = 10,
) -> dict[str, Any]:
    """Return streaming append/chunk latency for the full SPD configuration."""
    if batch_size <= 0 or warmup_ticks < 0 or measure_ticks <= 0 or euler_steps <= 0:
        raise ValueError("batch_size, measure_ticks, and euler_steps must be positive")
    checkpoint = Path(dino_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {checkpoint}; benchmark does not download weights"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency setup
        raise RuntimeError("policy benchmark requires PyTorch 2.11+") from exc
    from .policy import SPDPolicy, parameter_summary

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device is unavailable: {target}")
    torch.manual_seed(0)
    config = SPDModelConfig()
    model = SPDPolicy(config)
    missing, unexpected = model.load_dino(checkpoint)
    parameter_report = parameter_summary(model)
    model.set_dino_bfloat16(target.type == "cuda")
    model = model.to(target).eval()
    append_observation = model.append_observation
    sample_actions_cached = model.sample_actions_cached
    if compile_model:
        # Compile the two deployment methods explicitly. ``torch.compile(model)``
        # only wraps ``forward`` and would leave the rolling-KV path untouched.
        append_observation = torch.compile(append_observation, fullgraph=False, dynamic=True)
        sample_actions_cached = torch.compile(sample_actions_cached, fullgraph=False, dynamic=True)

    qpos = torch.zeros((batch_size, ROBOT_DOF), device=target)
    previous = torch.zeros_like(qpos)
    image = {
        camera: torch.zeros((batch_size, 3, 224, 224), device=target)
        for camera in CAMERA_NAMES
    }

    def append(cache: Any, step: int) -> Any:
        images = image if step % IMAGE_STRIDE == 0 else None
        return append_observation(cache, qpos, previous, step=step, images=images)

    with torch.inference_mode():
        cache = None
        for step in range(HISTORY_STEPS):
            cache = append(cache, step)
        for _ in range(warmup_ticks):
            step = cache.last_step + 1
            cache = append(cache, step)
            if step % IMAGE_STRIDE == 0:
                sample_actions_cached(cache, num_steps=euler_steps)
        _synchronize(torch, target)
        append_ms: list[float] = []
        chunk_ms: list[float] = []
        for _ in range(measure_ticks):
            step = cache.last_step + 1
            before = time.perf_counter_ns()
            cache = append(cache, step)
            _synchronize(torch, target)
            append_ms.append((time.perf_counter_ns() - before) / 1e6)
            if step % IMAGE_STRIDE == 0:
                before = time.perf_counter_ns()
                sample_actions_cached(cache, num_steps=euler_steps)
                _synchronize(torch, target)
                chunk_ms.append((time.perf_counter_ns() - before) / 1e6)

    import numpy as np

    def stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(np.max(array)),
        }

    append_stats = stats(append_ms)
    chunk_stats = stats(chunk_ms)
    budget = 1000.0 / 30.0
    return {
        "device": str(target),
        "batch_size": batch_size,
        "compile": bool(compile_model),
        "euler_steps": euler_steps,
        "dino_checkpoint": str(checkpoint),
        "dino_checkpoint_sha256": _sha256(checkpoint),
        "dino_missing_keys": len(missing),
        "dino_unexpected_keys": len(unexpected),
        "parameters": parameter_report,
        "append_observation": append_stats,
        "sample_actions_cached": chunk_stats,
        "control_budget_ms": budget,
        "chunk_deadline_p95_ok": bool(chunk_stats["count"] and float(chunk_stats["p95_ms"]) <= budget),
        "qualification": "diagnostic; pass --enforce-deadline only after target-GPU review",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-ticks", type=int, default=2)
    parser.add_argument("--measure-ticks", type=int, default=16)
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--enforce-deadline", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = benchmark_policy(
            args.dino_checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            warmup_ticks=args.warmup_ticks,
            measure_ticks=args.measure_ticks,
            compile_model=args.compile,
            euler_steps=args.euler_steps,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.enforce_deadline and not report["chunk_deadline_p95_ok"]:
        return 2
    return 0


__all__ = ["benchmark_policy", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
