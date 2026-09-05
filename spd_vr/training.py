"""Distributed training loop for the pure SPD 54-DoF policy."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import SPDTrainConfig, validate_spd_model_config
from .contracts import ROBOT_DOF
from .data import (
    SPDSequenceDataset,
    scan_episodes,
    validate_episode,
    validate_normalization,
)
from .policy import SPDPolicy, load_spd_checkpoint, parameter_summary


torch.set_float32_matmul_precision("high")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture every process-local RNG used by the training loop.

    Checkpoints are written by rank zero, so distributed callers gather one
    copy of this mapping per rank before serializing it.  CUDA state is
    optional for CPU-only smoke tests but is included whenever CUDA is
    available; silently dropping it would make a resumed run diverge at the
    first stochastic forward pass.
    """

    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`.

    Older checkpoints only contain ``torch`` and ``numpy``; those remain
    accepted for backwards compatibility.  A checkpoint that contains CUDA
    state cannot be resumed on a CPU-only process because doing so would make
    the stochastic stream unverifiable, so that mismatch fails closed.
    """

    if not isinstance(state, Mapping):
        raise ValueError("checkpoint RNG state must be a mapping")
    torch_state = state.get("torch")
    if torch_state is not None:
        if not isinstance(torch_state, torch.Tensor):
            raise ValueError("checkpoint torch RNG state must be a tensor")
        torch.set_rng_state(torch_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
            raise ValueError("checkpoint NumPy RNG state is malformed")
        np.random.set_state(numpy_state)
    python_state = state.get("python")
    if python_state is not None:
        if not isinstance(python_state, tuple) or len(python_state) != 3:
            raise ValueError("checkpoint Python RNG state is malformed")
        random.setstate(python_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if not isinstance(cuda_state, (list, tuple)) or not all(
            isinstance(value, torch.Tensor) for value in cuda_state
        ):
            raise ValueError("checkpoint CUDA RNG state is malformed")
        torch.cuda.set_rng_state_all(list(cuda_state))


def compute_normalization(root: str | Path) -> dict[str, list[float]]:
    """Compute stable per-joint statistics from actual MuJoCo qpos."""
    count = 0
    total = np.zeros(ROBOT_DOF, dtype=np.float64)
    squared = np.zeros(ROBOT_DOF, dtype=np.float64)
    for path in scan_episodes(root):
        validate_episode(path)
        with h5py.File(path, "r") as handle:
            dataset = handle["raw/observation/qpos"]
            training_index = handle["training/index_30hz"][:]
            contact_eligible = (
                handle["training/contact_eligible"][:]
                if "training/contact_eligible" in handle
                else np.ones(training_index.shape, dtype=np.bool_)
            )
            for start in range(0, len(training_index), 4096):
                rows = training_index[start : start + 4096]
                valid = np.all(handle["raw/validity/sides"][rows], axis=1) & contact_eligible[
                    start : start + len(rows)
                ]
                rows = rows[valid]
                if not len(rows):
                    continue
                value = np.asarray(dataset[rows], dtype=np.float64)
                total += value.sum(axis=0)
                squared += np.square(value).sum(axis=0)
                count += len(value)
    if count == 0:
        raise ValueError(f"no qpos rows found under {root}")
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return validate_normalization({
        "qpos_mean": mean.tolist(),
        "qpos_std": std.tolist(),
        # SPD actions are future actual qpos, so they share the exact statistic.
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
    })


class EMA:
    def __init__(self, model: torch.nn.Module, half_life_steps: float) -> None:
        if half_life_steps <= 0:
            raise ValueError("EMA half-life must be positive")
        self.decay = math.exp(math.log(0.5) / half_life_steps)
        self.state = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if value.requires_grad and value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.named_parameters():
            if name in self.state:
                self.state[name].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "model": self.state}


def split_muon_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Use Muon for trainable matrices and AdamW for vectors/scalars."""
    matrices, remainder = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (matrices if parameter.ndim == 2 else remainder).append(parameter)
    if not matrices or not remainder:
        raise ValueError("optimizer split unexpectedly produced an empty group")
    return matrices, remainder


def build_optimizers(
    model: torch.nn.Module, config: SPDTrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    matrices, remainder = split_muon_parameters(model)
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("SPD training requires a PyTorch build that provides torch.optim.Muon")
    muon = muon_cls(
        matrices,
        lr=config.optim.learning_rate,
        momentum=config.optim.muon_momentum,
        weight_decay=config.optim.weight_decay,
    )
    adamw = torch.optim.AdamW(
        remainder,
        lr=config.optim.learning_rate,
        betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        eps=config.optim.adam_epsilon,
        weight_decay=config.optim.weight_decay,
    )
    return muon, adamw


def optimization_step(
    model: torch.nn.Module,
    muon: torch.optim.Optimizer,
    adamw: torch.optim.Optimizer,
    ema: EMA,
    batch: Mapping[str, Any],
    *,
    max_grad_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one shared SPD optimization step and update EMA.

    Keeping this small seam public lets a deterministic tiny overfit test use
    the exact Muon/AdamW/EMA path used by the distributed loop.  It does not
    change the production model, data schema, or checkpoint format.
    """
    if isinstance(max_grad_norm, bool) or not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    model.train()
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    loss = model(batch)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("SPD training produced a non-finite scalar loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
    if not torch.isfinite(grad_norm):
        raise ValueError("SPD training produced a non-finite gradient norm")
    muon.step()
    adamw.step()
    ema.update(_module(model))
    return loss.detach(), torch.as_tensor(grad_norm).detach()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _module(model: torch.nn.Module) -> SPDPolicy:
    module = model.module if isinstance(model, DDP) else model
    return getattr(module, "_orig_mod", module)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    batches: int,
) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=device).manual_seed(0)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = _to_device(batch, device)
        noise = torch.randn(
            batch["future_action"].shape,
            device=device,
            dtype=batch["future_action"].dtype,
            generator=generator,
        )
        flow_time = torch.rand(
            batch["future_action"].shape[:2], device=device, generator=generator
        )
        losses.append(model(batch, noise=noise, flow_time=flow_time).detach())
    model.train()
    if not losses:
        return float("nan")
    result = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.AVG)
    return float(result.item())


def train(config: SPDTrainConfig) -> None:
    validate_spd_model_config(config.model)
    root = Path(config.dataset_root)
    train_root, val_root = root / "train", root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(f"expected HDF5 splits at {train_root} and {val_root}")
    dino_path = Path(config.dino_checkpoint)
    if not dino_path.is_file():
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {dino_path}; ABC checkpoints are intentionally unsupported"
        )
    dino_sha256 = _sha256(dino_path)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    norm_path = root / "normalization.json"
    if norm_path.exists():
        persisted = json.loads(norm_path.read_text(encoding="utf-8"))
        if persisted is None or (isinstance(persisted, dict) and not persisted):
            raise ValueError(
                f"{norm_path} must contain the complete train-only normalization mapping"
            )
        normalization = validate_normalization(persisted)
    else:
        normalization = compute_normalization(train_root)
        norm_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    symmetry_spec = None
    symmetry_spec_sha256 = None
    if config.symmetry_spec_path is not None:
        spec_path = Path(config.symmetry_spec_path)
        if not spec_path.is_file():
            raise FileNotFoundError(f"symmetry spec not found: {spec_path}")
        symmetry_spec_sha256 = _sha256(spec_path)
        raw_spec = spec_path.read_text(encoding="utf-8")
        if spec_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            document = yaml.safe_load(raw_spec)
        else:
            document = json.loads(raw_spec)
        from .augment import SymmetrySpec

        symmetry_spec = SymmetrySpec.from_mapping(document)
    if not 0.0 <= float(config.symmetry_probability) <= 1.0:
        raise ValueError("symmetry_probability must be in [0,1]")
    if config.symmetry_probability and symmetry_spec is None:
        raise ValueError("symmetry_spec_path is required for symmetry augmentation")
    if not 0.0 <= float(config.visual_randomization_probability) <= 1.0:
        raise ValueError("visual_randomization_probability must be in [0,1]")
    if not 0.0 <= float(config.visual_randomization_strength) <= 1.0:
        raise ValueError("visual_randomization_strength must be in [0,1]")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    train_dataset = SPDSequenceDataset(
        train_root,
        normalization=normalization,
        symmetry_spec=symmetry_spec,
        symmetry_probability=config.symmetry_probability,
        visual_randomization_probability=config.visual_randomization_probability,
        visual_randomization_strength=config.visual_randomization_strength,
    )
    val_dataset = SPDSequenceDataset(val_root, normalization=normalization)
    sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=config.seed, drop_last=True)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        drop_last=False,
    )

    model = SPDPolicy(config.model)
    missing, unexpected = model.load_dino(dino_path)
    model.set_dino_bfloat16(config.dino_bf16)
    model = model.to(device)
    muon, adamw = build_optimizers(model, config)
    ema = EMA(model, config.ema_half_life_steps)
    step = 0
    epoch = 0
    if config.resume is not None:
        checkpoint = load_spd_checkpoint(model, config.resume, use_ema=False)
        if "muon" not in checkpoint or "adamw" not in checkpoint or "ema" not in checkpoint:
            raise ValueError("resume checkpoint must contain model, EMA, Muon, and AdamW state")
        muon.load_state_dict(checkpoint["muon"])
        adamw.load_state_dict(checkpoint["adamw"])
        ema_state = checkpoint["ema"].get("model")
        if not isinstance(ema_state, dict) or set(ema_state) != set(ema.state):
            raise ValueError("resume EMA state does not match the trainable model")
        for name, value in ema_state.items():
            if tuple(value.shape) != tuple(ema.state[name].shape):
                raise ValueError(f"resume EMA shape mismatch: {name}")
            ema.state[name].copy_(value)
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if step < 0:
            raise ValueError("resume checkpoint step must be non-negative")
        if epoch < 0:
            raise ValueError("resume checkpoint epoch must be non-negative")
        rng_by_rank = checkpoint.get("rng_by_rank")
        if rng_by_rank is not None:
            expected_world_size = int(checkpoint.get("rng_world_size", len(rng_by_rank)))
            current_world_size = dist.get_world_size() if distributed else 1
            if (
                not isinstance(rng_by_rank, (list, tuple))
                or expected_world_size != current_world_size
                or len(rng_by_rank) != expected_world_size
                or rank >= len(rng_by_rank)
            ):
                raise ValueError("resume checkpoint does not contain RNG state for this rank")
            rng = rng_by_rank[rank]
        else:
            # Checkpoints written before per-rank capture remain loadable.
            rng = checkpoint.get("rng")
        if rng is not None:
            if not isinstance(rng, Mapping):
                raise ValueError("resume checkpoint RNG state is malformed")
            restore_rng_state(rng)
        if rank == 0:
            print(f"resumed SPD checkpoint {config.resume} at step={step}")
        checkpoint_dino_sha256 = checkpoint.get("dino_checkpoint_sha256")
        if checkpoint_dino_sha256 is not None and checkpoint_dino_sha256 != dino_sha256:
            raise ValueError(
                "resume DINOv3 checkpoint hash differs from the original training run"
            )
        checkpoint_symmetry_sha256 = checkpoint.get("symmetry_spec_sha256")
        if checkpoint_symmetry_sha256 is not None and checkpoint_symmetry_sha256 != symmetry_spec_sha256:
            raise ValueError("resume symmetry spec hash differs from the original training run")
    if config.compile:
        model = torch.compile(model, fullgraph=False)
    if distributed:
        model = DDP(model, device_ids=[device.index], static_graph=False)

    wandb = None
    if config.log_wandb and rank == 0:
        try:
            import wandb as wandb_module

            wandb = wandb_module
            wandb.init(project=config.wandb_project, config=asdict(config))
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    if rank == 0:
        print(
            f"DINOv3 loaded (missing={len(missing)}, unexpected={len(unexpected)}); "
            f"parameters={parameter_summary(_module(model))}"
        )
        print(f"train_windows={len(train_dataset)} val_windows={len(val_dataset)}")

    last_log = time.monotonic()
    model.train()
    while step < config.train_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= config.train_steps:
                break
            batch = _to_device(batch, device)
            loss, grad_norm = optimization_step(
                model,
                muon,
                adamw,
                ema,
                batch,
                max_grad_norm=config.optim.max_grad_norm,
            )
            step += 1

            if step % config.log_every == 0:
                reduced = loss.detach()
                if distributed:
                    dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
                if rank == 0:
                    elapsed = time.monotonic() - last_log
                    last_log = time.monotonic()
                    metrics = {
                        "loss": float(reduced.item()),
                        "grad_norm": float(grad_norm.item()),
                        "steps_per_second": config.log_every / elapsed,
                    }
                    print(f"step={step} " + " ".join(f"{k}={v:.5g}" for k, v in metrics.items()))
                    if wandb:
                        wandb.log(metrics, step=step)

            if step % config.val_every == 0:
                value = _validate(model, val_loader, device, config.val_batches)
                if rank == 0:
                    print(f"step={step} val_flow_loss={value:.6g}")
                    if wandb:
                        wandb.log({"val_flow_loss": value}, step=step)

            if step % config.ckpt_every == 0:
                # Every rank contributes its own stochastic stream.  The
                # all-gather is intentionally inside the checkpoint cadence,
                # so normal training still has no collective overhead beyond
                # the DDP gradient synchronizations.
                rank_rng = capture_rng_state()
                if distributed:
                    gathered_rng: list[Any] = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered_rng, rank_rng)
                else:
                    gathered_rng = [rank_rng]
                if rank == 0:
                    checkpoint = {
                        "model": {
                            name: value
                            for name, value in _module(model).state_dict().items()
                            if not name.startswith("img_backbone.")
                        },
                        "ema": ema.state_dict(),
                        "muon": muon.state_dict(),
                        "adamw": adamw.state_dict(),
                        "normalization": normalization,
                        "config": asdict(config),
                        "step": step,
                        "epoch": epoch,
                        "dino_checkpoint": str(dino_path.resolve()),
                        "dino_checkpoint_sha256": dino_sha256,
                        "symmetry_spec_sha256": symmetry_spec_sha256,
                        # Keep the rank-zero key for old tooling, while the
                        # per-rank list is what exact distributed resume uses.
                        "rng": gathered_rng[0],
                        "rng_by_rank": gathered_rng,
                        "rng_world_size": len(gathered_rng),
                    }
                    path = output / f"step_{step:07d}.pt"
                    torch.save(checkpoint, path)
                    torch.save(checkpoint, output / "last.pt")
                    print(f"saved {path}")
        epoch += 1
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    import tyro

    train(tyro.cli(SPDTrainConfig))


if __name__ == "__main__":
    main()


__all__ = [
    "EMA",
    "build_optimizers",
    "capture_rng_state",
    "compute_normalization",
    "optimization_step",
    "restore_rng_state",
    "split_muon_parameters",
    "train",
]
