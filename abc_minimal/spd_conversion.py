"""One-time, CPU-only migration of trusted spd-paired-kv-v2 training files.

This is a weights-only graph conversion, not an old-policy runtime adapter.
The new parameterization has different optimization dynamics: optimizer, RNG,
and scheduler state deliberately cannot be resumed across this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .checkpointing import checkpoint_step
from .config import SPDConfig
from .dino_weights import load_hf_dino_weights
from .spd import SPD_ARCHITECTURE, SPDPolicy, ObservationBlock, _validate_weights, validate_spd_config
from .tianji_data import validate_spd_norm_stats

SOURCE_ARCHITECTURE = "spd-paired-kv-v2"
# Official DINOv3 ViT-B/16 artifact used by the supported 10k training run.
OFFICIAL_DINO_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
_RENAMES = {"state_embedding": "qpos_embedding", "previous_actions_embedding": "previous_action_embedding"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_config(source: Mapping) -> SPDConfig:
    if not isinstance(source, Mapping) or source.get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError(f"source architecture must be {SOURCE_ARCHITECTURE!r}")
    training = source.get("config")
    saved = training.get("model") if isinstance(training, Mapping) else None
    expected = asdict(SPDConfig())
    fields = (set(expected) - {"vision_pool_num_queries"}) | {"vision_queries"}
    if not isinstance(saved, Mapping) or set(saved) != fields:
        raise ValueError("source config.model fields differ from supported SPDModelConfig")
    converted = dict(saved)
    converted["vision_pool_num_queries"] = converted.pop("vision_queries")
    cameras = converted["camera_keys"]
    if not isinstance(cameras, (tuple, list)):
        raise ValueError("source camera_keys must be an ordered sequence")
    converted["camera_keys"] = tuple(cameras)
    config = SPDConfig(**converted)
    errors = validate_spd_config(config)
    if errors:
        raise ValueError("invalid source config: " + "; ".join(errors))
    # Only the measured production graph is supported, including 54/256/8/8.
    # Noise remains a training-only perturbation of normalized 54-D histories.
    for name, value in expected.items():
        if name not in {"dino_frame_batch_size", "observation_noise_std", "action_noise_std"}:
            if getattr(config, name) != value:
                raise ValueError(f"unsupported source graph: {name}={getattr(config, name)!r}, expected {value!r}")
    return config


def _convert_normalization(source: Mapping) -> dict:
    fields = {f"{prefix}_{stat}" for prefix in ("qpos", "action") for stat in ("mean", "std")}
    if not isinstance(source, Mapping) or set(source) != fields:
        raise ValueError("source normalization must contain exactly qpos/action mean/std")
    vectors = {}
    for name in fields:
        try:
            raw = np.asarray(source[name])
            if raw.dtype.kind not in "fiu" or raw.shape != (54,):
                raise ValueError("not a numeric 54-vector")
            numeric = raw.astype(np.float64)
            if not np.isfinite(numeric).all() or np.any(np.abs(numeric) > np.finfo(np.float32).max):
                raise ValueError("not finite in float32")
            vector = numeric.astype(np.float32)
            if name.endswith("_std") and np.any(vector <= 0):
                raise ValueError("source standard deviation must be positive in float32")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid source normalization {name}: {exc}") from exc
        vectors[name] = vector
    epsilon = np.float32(1e-6)
    smallest = np.nextafter(np.float32(0), np.float32(1))
    result = {}
    for old, new in (("qpos", "state"), ("action", "actions")):
        effective = np.maximum(vectors[old + "_std"], epsilon)
        target_std = effective - epsilon
        target_std[target_std == 0] = smallest
        reconstructed = target_std + epsilon
        # Positive IEEE float32 bit patterns are monotonically ordered. This
        # also checks the largest finite float without overflowing nextafter.
        ulps = np.abs(reconstructed.view(np.uint32).astype(np.int64) - effective.view(np.uint32).astype(np.int64))
        if not np.isfinite(reconstructed).all() or np.any(ulps > 1):
            raise ValueError(f"{old} normalization denominator cannot be preserved within one float32 ULP")
        result[new] = {"mean": vectors[old + "_mean"].tolist(), "std": target_std.tolist()}
    return validate_spd_norm_stats(result)


def _old_name(name: str) -> str:
    root, separator, suffix = name.partition(".")
    return _RENAMES.get(root, root) + separator + suffix


def _old_reattention_shapes(width: int) -> dict[str, tuple[int, ...]]:
    return {
        "query.weight": (width, width), "query.bias": (width,),
        "output.weight": (width, width), "output.bias": (width,),
        "norm.weight": (width,), "norm.bias": (width,),
        "attention.in_proj_weight": (3 * width, width),
        "attention.in_proj_bias": (3 * width,),
        "attention.out_proj.weight": (width, width),
        "attention.out_proj.bias": (width,),
    }


def _fold_reattention(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold the old Linear -> MHA -> Linear, retaining its normalization.

    Equal hidden/vision widths and head counts are essential: otherwise the
    attention softmax's head dimension and scale would change. All products
    and bias sums are computed in float64 before a single cast back to FP32.
    """
    projection = source["attention.in_proj_weight"]
    bias = source["attention.in_proj_bias"]
    width = projection.shape[1]
    query = projection[:width].double()
    output = source["output.weight"].double()
    return {
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
        "attention.in_proj_weight": torch.cat((
            (query @ source["query.weight"].double()).to(projection.dtype),
            projection[width:],
        )),
        "attention.in_proj_bias": torch.cat((
            (query @ source["query.bias"].double() + bias[:width].double()).to(bias.dtype),
            bias[width:],
        )),
        "attention.out_proj.weight": (output @ source["attention.out_proj.weight"].double()).to(projection.dtype),
        "attention.out_proj.bias": (output @ source["attention.out_proj.bias"].double() + source["output.bias"].double()).to(bias.dtype),
    }


def _convert_weights(source: Mapping, model: SPDPolicy, *, ema: bool = False) -> dict:
    config = model.config
    if config.hidden_size != config.vit_embed_dim:
        raise ValueError("cannot fold reattention with unequal hidden/vision widths")
    target = (
        {name: value for name, value in model.named_parameters() if value.requires_grad}
        if ema else {name: value for name, value in model.state_dict().items() if not name.startswith("img_backbone.")}
    )
    expected = {_old_name(name): value for name, value in target.items() if not name.startswith("vision_reattention.")}
    with torch.device("meta"):
        # The old terminal block existed, but only its normalized-input K/V
        # had a consumer. Require ALL of its dead tensors before dropping them.
        terminal = ObservationBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
        for name, value in terminal.state_dict().items():
            expected[f"observation_blocks.{config.depth - 1}.{name}"] = value
        for stage in range(config.depth // 2):
            for camera in config.camera_keys:
                for name, shape in _old_reattention_shapes(config.hidden_size).items():
                    expected[f"vision_reattention.{stage}.{camera}.{name}"] = torch.empty(shape)
    _validate_weights(source, expected, "source EMA" if ema else "source model")
    # The supported source is FP32. Refuse an implicit precision change even
    # for unused tensors, rather than accepting a different graph artifact.
    if any(value.dtype != torch.float32 for value in source.values()):
        raise ValueError("source weights and persistent buffers must be float32")
    converted = {name: source[_old_name(name)] for name in target if not name.startswith("vision_reattention.")}
    for stage in range((config.depth - 1) // 2):
        for camera in config.camera_keys:
            prefix = f"vision_reattention.{stage}.{camera}."
            local = {name: source[prefix + name] for name in _old_reattention_shapes(config.hidden_size)}
            converted.update({prefix + name: value for name, value in _fold_reattention(local).items()})
    _validate_weights(converted, target, "converted EMA" if ema else "converted model")
    return converted


def _source_cameras(source: Mapping, config: SPDConfig) -> list[str]:
    contract = source.get("dataset_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("source requires dataset_contract camera provenance")
    cameras = contract.get("observed_camera_keys")
    if not isinstance(cameras, (list, tuple)) or not cameras or any(not isinstance(camera, str) for camera in cameras):
        raise ValueError("source observed_camera_keys must be a nonempty ordered camera sequence")
    if len(set(cameras)) != len(cameras) or not set(cameras) <= set(config.camera_keys):
        raise ValueError("source observed_camera_keys contain duplicate or unknown cameras")
    expected_missing = [camera for camera in config.camera_keys if camera not in cameras]
    dataset = contract.get("config")
    if (contract.get("model_camera_keys") not in (list(config.camera_keys), tuple(config.camera_keys))
            or contract.get("unavailable_camera_keys") not in (expected_missing, tuple(expected_missing))
            or not isinstance(dataset, Mapping)
            or dataset.get("camera_names") not in (list(cameras), tuple(cameras))):
        raise ValueError("source dataset camera provenance is inconsistent")
    return list(cameras)


def _convert_payload(source: Mapping, model: SPDPolicy, *, dino_sha256: str, provenance: dict) -> dict:
    """Assemble the clean checkpoint; kept separate for small graph fixtures."""
    if not isinstance(source, Mapping) or source.get("architecture") != SOURCE_ARCHITECTURE:
        raise ValueError(f"source architecture must be {SOURCE_ARCHITECTURE!r}")
    if source.get("dino_checkpoint_sha256") != dino_sha256:
        raise ValueError("source DINO hash differs from supplied official DINO checkpoint")
    step = source.get("step")
    if isinstance(step, bool) or not isinstance(step, Integral) or step <= 0:
        raise ValueError("source step must be a positive integer")
    if any(name in source and source[name] != step for name in ("global_step", "training_step")):
        raise ValueError("source checkpoint step metadata is inconsistent")
    ema = source.get("ema")
    if not isinstance(ema, Mapping) or set(ema) != {"model", "decay"}:
        raise ValueError("source requires EMA model and decay")
    decay = ema["decay"]
    if isinstance(decay, bool) or not isinstance(decay, Real) or not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("source EMA decay must be finite and in [0,1]")
    norm_stats = _convert_normalization(source.get("normalization"))
    cameras = _source_cameras(source, model.config)
    raw = _convert_weights(source.get("model"), model)
    averaged = _convert_weights(ema["model"], model, ema=True)
    return {
        "policy": "spd", "architecture": SPD_ARCHITECTURE,
        "model_config": asdict(model.config), "model": raw,
        "ema": {"decay": float(decay), "model": averaged},
        "norm_stats": norm_stats, "dino_sha256": dino_sha256,
        "global_step": checkpoint_step(source), "weights_only": True,
        "exact_resume": False, "source_camera_keys": cameras,
        "source_provenance": {
            **provenance, "architecture": SOURCE_ARCHITECTURE,
            "global_step": int(step), "model_config": dict(source["config"]["model"]),
            "conversion": "fp64-reattention-fold-v1",
            "normalization": "float32 max(std,1e-6) -> std+1e-6; denominator within 1 ULP",
            "training_state": "not migrated; model-only finetuning, never exact resume",
        },
    }


def _atomic_save(checkpoint: Mapping, output: Path) -> None:
    """Publish a completely flushed file atomically, without replacing any path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() is atomic AND fails if another writer wins.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)


def convert_spd_checkpoint(source_path: str | Path, output_path: str | Path, dino_checkpoint: str | Path) -> dict:
    """Convert a trusted legacy training checkpoint, returning a JSON-safe report.

    Training checkpoints contain pickle objects (including RNG metadata); only
    pass trusted files. The output is slim: official frozen DINO remains an
    explicitly hash-checked external dependency. No CUDA device is used.
    """
    source_path = Path(source_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().absolute()
    dino_path = Path(dino_checkpoint).expanduser().resolve(strict=True)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output}")
    if dino_path.suffix != ".safetensors" or _sha256(dino_path) != OFFICIAL_DINO_SHA256:
        raise ValueError("DINO must be the supported official ViT-B/16 safetensors checkpoint (SHA256 mismatch)")
    # mmap avoids materializing the unused optimizer states from the large
    # training file; nothing from them is carried into the clean checkpoint.
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    config = _source_config(source)
    if source.get("dino_checkpoint_sha256") != OFFICIAL_DINO_SHA256:
        raise ValueError("source DINO hash differs from supplied official DINO checkpoint")
    with torch.device("meta"):
        model = SPDPolicy(config)
    dino_weights = load_hf_dino_weights(dino_path, config)
    _validate_weights(dino_weights, model.img_backbone.dinov3_model.state_dict(), "official DINO")
    del dino_weights
    source_sha256 = _sha256(source_path)
    checkpoint = _convert_payload(source, model, dino_sha256=OFFICIAL_DINO_SHA256, provenance={
        "path": str(source_path), "sha256": source_sha256,
        "dino_checkpoint": str(dino_path),
    })
    _atomic_save(checkpoint, output)
    return {
        "output_path": str(output), "architecture": SPD_ARCHITECTURE,
        "source_sha256": source_sha256, "dino_sha256": OFFICIAL_DINO_SHA256,
        "global_step": checkpoint["global_step"], "weights_only": True,
        "exact_resume": False, "source_camera_keys": checkpoint["source_camera_keys"],
        "model_tensors": len(checkpoint["model"]), "ema_tensors": len(checkpoint["ema"]["model"]),
    }
