"""Configuration for the SPD policy and Tianji-Wuji2 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from pathlib import Path

from abc_minimal.config import default_cache_root

from .contracts import ACTION_CHUNK, CAMERA_NAMES, HISTORY_STEPS, IMAGE_STRIDE, ROBOT_DOF


@dataclass
class SPDModelConfig:
    hidden_size: int = 768
    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    state_dim: int = ROBOT_DOF
    action_dim: int = ROBOT_DOF
    history_steps: int = HISTORY_STEPS
    chunk_length: int = ACTION_CHUNK
    image_stride: int = IMAGE_STRIDE
    attention_window_steps: int = 32
    camera_keys: tuple[str, ...] = CAMERA_NAMES
    vision_queries: int = 4
    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vision_pool_num_heads: int = 8
    vision_pool_mlp_ratio: int = 4
    observation_noise_std: float = 0.03
    action_noise_std: float = 0.03


@dataclass
class SPDOptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0


@dataclass
class SPDTrainConfig:
    dataset_root: str = field(
        default_factory=lambda: str(default_cache_root() / "spd")
    )
    output_dir: str = field(
        default_factory=lambda: str(default_cache_root() / "spd_checkpoints")
    )
    resume: str | None = None
    seed: int = 123
    batch_size: int = 64
    num_workers: int = 8
    train_steps: int = 170_000
    log_every: int = 20
    ckpt_every: int = 5_000
    val_every: int = 2_500
    val_batches: int = 4
    ema_half_life_steps: float = 20.0
    dino_checkpoint: str = field(
        default_factory=lambda: str(
            default_cache_root() / "dinov3_vitb16_pretrain_lvd1689m.pth"
        )
    )
    compile: bool = True
    dino_bf16: bool = True
    log_wandb: bool = False
    wandb_project: str = "spd-vr"
    symmetry_probability: float = 0.0
    symmetry_spec_path: str | None = None
    visual_randomization_probability: float = 0.0
    visual_randomization_strength: float = 0.65
    model: SPDModelConfig = field(default_factory=SPDModelConfig)
    optim: SPDOptimConfig = field(default_factory=SPDOptimConfig)


@dataclass
class TeleopConfig:
    urdf_path: str = str(
        Path(__file__).resolve().parents[1]
        / "assets"
        / "tianji_wuji2"
        / "tianji_wuji2.urdf"
    )
    control_hz: int = 60
    physics_hz: int = 480
    stale_after_ms: float = 50.0
    alignment_frames: int = 10
    alignment_position_std_m: float = 0.005
    alignment_quaternion_dot_min: float = 0.995

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.physics_hz <= 0:
            raise ValueError("control_hz and physics_hz must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if not math.isfinite(float(self.stale_after_ms)) or self.stale_after_ms <= 0:
            raise ValueError("stale_after_ms must be finite and positive")
        if self.alignment_frames <= 0:
            raise ValueError("alignment_frames must be positive")
        if (
            not math.isfinite(float(self.alignment_position_std_m))
            or self.alignment_position_std_m < 0
            or not math.isfinite(float(self.alignment_quaternion_dot_min))
            or not 0 < self.alignment_quaternion_dot_min <= 1
        ):
            raise ValueError("alignment thresholds are invalid")


def validate_spd_model_config(config: SPDModelConfig) -> None:
    if not isinstance(config, SPDModelConfig):
        raise TypeError("config must be an SPDModelConfig")
    integer_fields = (
        "hidden_size",
        "depth",
        "num_heads",
        "state_dim",
        "action_dim",
        "history_steps",
        "chunk_length",
        "image_stride",
        "attention_window_steps",
        "vision_queries",
        "vit_embed_dim",
        "vit_depth",
        "vit_num_heads",
        "vision_pool_num_heads",
        "vision_pool_mlp_ratio",
    )
    if any(
        isinstance(getattr(config, name), bool)
        or not isinstance(getattr(config, name), Integral)
        or int(getattr(config, name)) <= 0
        for name in integer_fields
    ):
        raise ValueError("SPD model dimensions must be positive integers")
    if any(
        not math.isfinite(float(getattr(config, name)))
        or float(getattr(config, name)) <= 0.0
        for name in ("mlp_ratio",)
    ):
        raise ValueError("mlp_ratio must be finite and positive")
    if any(
        not math.isfinite(float(getattr(config, name)))
        or float(getattr(config, name)) < 0.0
        for name in ("observation_noise_std", "action_noise_std")
    ):
        raise ValueError("observation/action noise std must be finite and non-negative")
    if config.state_dim != ROBOT_DOF or config.action_dim != ROBOT_DOF:
        raise ValueError("SPD Tianji-Wuji2 state/action dimensions must both be 54")
    if config.history_steps != HISTORY_STEPS or config.chunk_length != ACTION_CHUNK:
        raise ValueError("SPD history/chunk must be 256/8")
    if config.history_steps % config.image_stride:
        raise ValueError("history_steps must be divisible by image_stride")
    if config.image_stride != IMAGE_STRIDE:
        raise ValueError("SPD image_stride must be the fixed 8-step contract")
    if config.attention_window_steps <= 0 or config.attention_window_steps > config.history_steps:
        raise ValueError("attention_window_steps must be in [1, history_steps]")
    if tuple(config.camera_keys) != CAMERA_NAMES:
        raise ValueError(f"camera_keys must be {CAMERA_NAMES}")
    if config.hidden_size % config.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    if (config.hidden_size // config.num_heads) % 2:
        raise ValueError("attention head width must be even for RoPE")
    if config.hidden_size % 2:
        raise ValueError("hidden_size must be even for sinusoidal embeddings")
    if config.depth <= 0 or config.depth % 2:
        raise ValueError("depth must be a positive even number")
    if config.vit_embed_dim % config.vit_num_heads:
        raise ValueError("vit_embed_dim must be divisible by vit_num_heads")
    if config.vit_embed_dim % config.vision_pool_num_heads:
        raise ValueError("vit_embed_dim must be divisible by vision_pool_num_heads")


__all__ = [
    "SPDModelConfig",
    "SPDOptimConfig",
    "SPDTrainConfig",
    "TeleopConfig",
    "validate_spd_model_config",
]
