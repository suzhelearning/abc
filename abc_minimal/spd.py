"""SPD paired observation/action experts using ABC's frozen DINO and pooling.

Implementation choices where the paper leaves sharing unspecified: independent
observation/action blocks; shared initial attention pool with camera queries;
independent per-camera reattention after observation blocks 2/4/6. Reattention
has only its ordinary attention projections, not additional outer linears. The
last observation level projects normalized K/V only: its Q/O/FFN and the final
vision refresh have no consumer and are absent, not parameter-count padding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SPDConfig
from .dit import AttentionPoolBlock, DinoVisionBackbone, get_1d_sincos_pos_embed

SPD_ARCHITECTURE = "abc-spd-v1"


def validate_spd_config(config: SPDConfig) -> list[str]:
    """Return errors without constructing the potentially large model."""
    errors = []
    positive_ints = (
        "hidden_size", "depth", "num_heads", "state_dim", "action_dim",
        "history_steps", "chunk_length", "image_stride", "attention_window_steps",
        "vit_embed_dim", "vit_depth", "vit_num_heads", "vision_pool_num_queries",
        "vision_pool_num_heads", "vision_pool_mlp_ratio", "dino_frame_batch_size",
    )
    for name in positive_ints:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            errors.append(f"{name} must be a positive integer")
    if errors:
        return errors
    if config.hidden_size % config.num_heads or (config.hidden_size // config.num_heads) % 2:
        errors.append("hidden_size / num_heads must be an even integer for temporal RoPE")
    if config.vit_embed_dim % config.vit_num_heads or (config.vit_embed_dim // config.vit_num_heads) % 4:
        errors.append("vit_embed_dim / vit_num_heads must be divisible by four for DINO RoPE")
    if config.vit_embed_dim % config.vision_pool_num_heads or config.hidden_size % config.vision_pool_num_heads:
        errors.append("vision pool heads must divide vision and hidden widths")
    if config.history_steps % config.image_stride:
        errors.append("history_steps must be divisible by image_stride")
    if config.state_dim != 54 or config.action_dim != 54:
        errors.append("state_dim and action_dim must both be 54 for Tianji/Wuji2")
    if tuple(config.camera_keys) != ("top", "left_wrist", "right_wrist"):
        errors.append("camera_keys must be ordered top, left_wrist, right_wrist")
    if not isinstance(config.mlp_ratio, Real) or not math.isfinite(config.mlp_ratio) or config.mlp_ratio <= 0 or int(config.hidden_size * config.mlp_ratio) < 1:
        errors.append("mlp_ratio must produce a positive finite MLP width")
    for name in ("observation_noise_std", "action_noise_std"):
        value = getattr(config, name)
        if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            errors.append(f"{name} must be finite and non-negative")
    return errors


def _validate_weights(source: Mapping, expected: Mapping, label: str) -> None:
    if not isinstance(source, Mapping):
        raise ValueError(f"{label} must be a tensor mapping")
    missing, unexpected = set(expected) - set(source), set(source) - set(expected)
    if missing or unexpected:
        raise ValueError(f"{label} keys differ: missing={sorted(missing)} unexpected={sorted(unexpected, key=str)}")
    for name, value in source.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"{label} shape mismatch: {name}")
        if value.is_floating_point() != expected[name].is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{label} invalid or nonfinite tensor: {name}")


class FlowTimeEmbedding(nn.Module):
    """Gaussian Fourier features followed by a small MLP."""

    def __init__(self, hidden_size: int, fourier_size: int = 256) -> None:
        super().__init__()
        if fourier_size % 2:
            raise ValueError("fourier_size must be even")
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "frequencies",
            torch.randn(fourier_size // 2, generator=generator) * (2.0 * math.pi),
            persistent=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(fourier_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time[..., None].float() * self.frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.mlp(encoded.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, int(width * ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(width * ratio), width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TemporalKeyValue(nn.Module):
    """Normalized-input K/V projection with timestep RoPE on keys."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("attention head width must be even")
        self.heads = heads
        self.head_width = width // heads
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(self.head_width // 2, dtype=torch.float32)
            / (self.head_width // 2)
        )
        self.register_buffer("frequency", frequency, persistent=False)

    def _rope(self, value: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        angle = times.float()[:, None] * self.frequency[None, :]
        sin = torch.sin(torch.cat((angle, angle), dim=-1))[None, None]
        cos = torch.cos(torch.cat((angle, angle), dim=-1))[None, None]
        sin = sin.to(device=value.device, dtype=value.dtype)
        cos = cos.to(device=value.device, dtype=value.dtype)
        return value * cos + _rotate_half(value) * sin

    def project_key_value(
        self, key_value: torch.Tensor, key_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, key_length, _ = key_value.shape
        k = self.key(key_value).reshape(batch, key_length, self.heads, self.head_width).transpose(1, 2)
        v = self.value(key_value).reshape(batch, key_length, self.heads, self.head_width).transpose(1, 2)
        return self._rope(k, key_times), v

class TemporalAttention(TemporalKeyValue):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__(width, heads)
        self.query = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def project_query(
        self, query: torch.Tensor, query_times: torch.Tensor
    ) -> torch.Tensor:
        batch, query_length, _ = query.shape
        projected = self.query(query).reshape(
            batch, query_length, self.heads, self.head_width
        ).transpose(1, 2)
        return self._rope(projected, query_times)

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, query_length, _ = query.shape
        width = self.heads * self.head_width
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        else:
            raise ValueError("attention mask must be [Q,K] or [B,Q,K]")
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=~mask)
        attended = attended.transpose(1, 2).reshape(batch, query_length, width)
        return self.output(attended)


class ObservationKV(nn.Module):
    """Terminal observation level: no unconsumed Q/O, FFN, or normalization."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = TemporalKeyValue(width, heads)


class ObservationBlock(nn.Module):
    def __init__(self, width: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = TemporalAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = FeedForward(width, ratio)

    def update(
        self, value: torch.Tensor, normalized: torch.Tensor,
        times: torch.Tensor, layer: SPDLayerKV, mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.attention.project_query(normalized, times)
        value = value + self.attention.attend_projected(query, layer.key, layer.value, mask)
        return value + self.mlp(self.norm_mlp(value))


class ActionExpertBlock(nn.Module):
    """Action queries use the paired observation expert's projected K/V directly."""

    def __init__(self, width: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = TemporalAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = FeedForward(width, ratio)

    def forward(
        self, action: torch.Tensor, observation: SPDLayerKV,
        action_times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_attention(action)
        query = self.attention.project_query(normalized, action_times)
        action_key, action_value = self.attention.project_key_value(normalized, action_times)
        key = torch.cat((observation.key, action_key), dim=2)
        value = torch.cat((observation.value, action_value), dim=2)
        action = action + self.attention.attend_projected(query, key, value, mask)
        return action + self.mlp(self.norm_mlp(action))


class VisionReattention(nn.Module):
    """Ordinary cross-attention; no extra query/output projection wrappers."""

    def __init__(self, hidden_size: int, vision_size: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, kdim=vision_size, vdim=vision_size, batch_first=True,
        )

    def forward(self, pooled: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        refreshed, _ = self.attention(self.norm(pooled), raw, raw, need_weights=False)
        return pooled + refreshed


@dataclass(frozen=True, slots=True)
class SPDLayerKV:
    """Projected, normalized layer-input observation K/V and key metadata."""

    key: torch.Tensor
    value: torch.Tensor
    times: torch.Tensor
    validity: torch.Tensor


@dataclass(frozen=True, slots=True)
class SPDObservationCache:
    """Shared diffusion conditioning: full encoding or rolling 32-step K/V.

    Neither final observation memory nor action-specific observation projections
    are retained. Full and streaming encoders produce the same layer structure.
    """

    layers: tuple[SPDLayerKV, ...]
    last_step: int


def _causal_window_mask(
    query_times: torch.Tensor,
    key_times: torch.Tensor,
    window: int,
) -> torch.Tensor:
    delta = query_times[:, None] - key_times[None, :]
    allowed = (delta >= 0) & (delta < window)
    return ~allowed


def _action_context_mask(
    action_times: torch.Tensor,
    observation_times: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Prefix-parallel mask: history plus only the query's own action chunk."""
    observation_delta = action_times[:, None] - observation_times[None, :]
    observation_allowed = (observation_delta >= 0) & (observation_delta < window)
    same_chunk = action_times[:, None] == action_times[None, :]
    return ~torch.cat((observation_allowed, same_chunk), dim=1)


class SPDPolicy(nn.Module):
    """Long-context SPD policy with a frozen DINOv3 ViT-B/16 backbone."""

    def __init__(self, config: SPDConfig, vision_backbone: nn.Module | None = None) -> None:
        super().__init__()
        errors = validate_spd_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        self.config = config
        hidden = config.hidden_size
        self.camera_keys = tuple(config.camera_keys)
        self.chunk_length = config.chunk_length
        self.action_dim = config.action_dim
        self.image_steps = config.history_steps // config.image_stride

        self.state_embedding = nn.Linear(config.state_dim, hidden)
        self.previous_actions_embedding = nn.Linear(config.action_dim, hidden)
        self.action_embedding = nn.Linear(config.action_dim, hidden)
        self.token_type = nn.Embedding(4, hidden)
        self.register_buffer("chunk_position", torch.from_numpy(get_1d_sincos_pos_embed(hidden, config.chunk_length)).float(), persistent=True)
        self.chunk_position_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow_time = FlowTimeEmbedding(hidden)

        self.img_backbone = vision_backbone if vision_backbone is not None else DinoVisionBackbone(config)
        self.img_backbone.requires_grad_(False)
        self.img_backbone.eval()
        self.vision_queries = nn.ParameterDict(
            {
                camera: nn.Parameter(torch.randn(1, config.vision_pool_num_queries, config.vit_embed_dim) * 0.02)
                for camera in self.camera_keys
            }
        )
        # Shared initial pooling; camera-specific queries and later reattention.
        self.vision_pool = AttentionPoolBlock(
            config.vit_embed_dim,
            config.vision_pool_num_heads,
            config.vision_pool_mlp_ratio,
        )
        self.vision_projection = nn.Linear(config.vit_embed_dim, hidden)
        self.camera_embedding = nn.Embedding(len(self.camera_keys), hidden)

        self.observation_blocks = nn.ModuleList(
            (ObservationBlock(hidden, config.num_heads, config.mlp_ratio)
             if index < config.depth - 1 else ObservationKV(hidden, config.num_heads))
            for index in range(config.depth)
        )
        self.vision_reattention = nn.ModuleList(
            nn.ModuleDict({
                camera: VisionReattention(hidden, config.vit_embed_dim, config.vision_pool_num_heads)
                for camera in self.camera_keys
            })
            for _ in range((config.depth - 1) // 2)
        )
        self.action_expert = nn.ModuleList(
            ActionExpertBlock(hidden, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.action_norm = nn.LayerNorm(hidden)
        self.velocity_head = nn.Linear(hidden, config.action_dim)

        state_times = torch.arange(self.config.history_steps).repeat_interleave(2)
        image_times = torch.arange(0, self.config.history_steps, self.config.image_stride).repeat_interleave(
            len(self.camera_keys) * config.vision_pool_num_queries
        )
        observation_times = torch.cat((state_times, image_times))
        action_times = torch.arange(0, self.config.history_steps, self.config.image_stride).repeat_interleave(self.chunk_length)
        self.register_buffer("observation_times", observation_times, persistent=False)
        self.register_buffer("action_times", action_times, persistent=False)

    def train(self, mode: bool = True) -> SPDPolicy:
        super().train(mode)
        self.img_backbone.eval()
        return self

    def set_dino_bfloat16(self, enabled: bool = True) -> None:
        setter = getattr(self.img_backbone, "set_bfloat16", None)
        if setter is not None:
            setter(enabled)

    def load_dino(self, checkpoint: str | Path) -> tuple[list[str], list[str]]:
        checkpoint = Path(checkpoint)
        if checkpoint.suffix == ".safetensors":
            from .dino_weights import load_hf_dino_weights

            state = load_hf_dino_weights(checkpoint, self.config)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("DINOv3 checkpoint must contain a mapping")
        state = state.get("model", state)
        if not isinstance(state, Mapping) or not state:
            raise ValueError("DINOv3 checkpoint contains no model weights")
        cleaned = {}
        for key, value in state.items():
            if not isinstance(key, str):
                raise ValueError("DINOv3 checkpoint keys must be strings")
            # DDP/checkpoint wrappers can be nested.  Peel known wrappers one
            # at a time until the parameter name reaches the DINO module;
            # never slice an already-cleaned name a second time.
            while True:
                for prefix in ("module.", "backbone.", "dinov3_model."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                else:
                    break
            if not key:
                raise ValueError("DINOv3 checkpoint contains an empty parameter name")
            if key in cleaned:
                raise ValueError(f"DINOv3 checkpoint contains duplicate parameter: {key}")
            cleaned[key] = value
        target = getattr(self.img_backbone, "dinov3_model", self.img_backbone)
        _validate_weights(cleaned, target.state_dict(), "DINO checkpoint")
        result = target.load_state_dict(cleaned, strict=True)
        self.img_backbone.requires_grad_(False)
        self.img_backbone.eval()
        return list(result.missing_keys), list(result.unexpected_keys)

    def _vision_tokens(
        self, images: dict[str, torch.Tensor], camera_validity: torch.Tensor | None,
        *, batch_size: int, image_steps: int, device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        if not isinstance(images, dict) or set(images) - set(self.camera_keys):
            raise ValueError(f"images must use camera keys from {self.camera_keys}")
        shape = (batch_size, image_steps, len(self.camera_keys))
        if camera_validity is None:
            if set(images) != set(self.camera_keys):
                raise ValueError(f"images must contain exactly {self.camera_keys} without camera_validity")
            validity = torch.ones(shape, dtype=torch.bool, device=device)
        else:
            if (
                not isinstance(camera_validity, torch.Tensor)
                or camera_validity.dtype != torch.bool
                or camera_validity.shape != shape
                or camera_validity.device != device
            ):
                raise ValueError(f"camera_validity must be bool {shape} on the state device")
            validity = camera_validity
        projected_by_camera, raw_by_camera = [], {}
        for camera_index, camera in enumerate(self.camera_keys):
            indices = validity[:, :, camera_index].flatten().nonzero(as_tuple=True)[0]
            pooled = self.vision_projection.weight.new_zeros(
                (batch_size * image_steps, self.config.vision_pool_num_queries, self.config.hidden_size)
            )
            if camera not in images:
                if indices.numel():
                    raise ValueError(f"missing {camera} images have valid camera_validity entries")
            else:
                image = images[camera]
                if (
                    not isinstance(image, torch.Tensor) or image.ndim != 5
                    or image.shape[:3] != (batch_size, image_steps, 3)
                    or min(image.shape[3:]) <= 0 or not image.is_floating_point()
                    or image.device != device
                ):
                    raise ValueError(f"{camera} images must be floating [B,T,3,H,W] on the state device")
                if indices.numel():
                    flat_image = image.flatten(0, 1)
                    frame_batch_size = self.config.dino_frame_batch_size
                    # Select only valid frames, in bounded batches. Missing views
                    # never invoke DINO, even when callers supply placeholders.
                    with torch.no_grad():
                        raw = None
                        for start in range(0, indices.numel(), frame_batch_size):
                            frames = flat_image.index_select(0, indices[start:start + frame_batch_size])
                            if not torch.isfinite(frames).all():
                                raise ValueError(f"{camera} valid images must contain finite values")
                            tokens = self.img_backbone.encode_image_tokens(frames).to(self.vision_queries[camera].dtype)
                            if raw is None:
                                raw = tokens.new_empty((indices.numel(), *tokens.shape[1:]))
                            raw[start:start + tokens.shape[0]].copy_(tokens)
                    query = self.vision_queries[camera].expand(indices.numel(), -1, -1)
                    valid_pooled = self.vision_projection(self.vision_pool(raw, query))
                    valid_pooled = valid_pooled + self.camera_embedding.weight[camera_index] + self.token_type.weight[2]
                    pooled = pooled.to(valid_pooled.dtype).index_copy(0, indices, valid_pooled)
                    raw_by_camera[camera] = (indices, raw)
            projected_by_camera.append(pooled.reshape(batch_size, image_steps, self.config.vision_pool_num_queries, -1))
        return torch.stack(projected_by_camera, dim=2), raw_by_camera, validity

    def _refresh_vision(
        self, index: int, observation: torch.Tensor, vision_start: int,
        image_steps: int, raw_by_camera: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch = observation.shape[0]
        pooled = observation[:, vision_start:].reshape(
            batch, image_steps, len(self.camera_keys), self.config.vision_pool_num_queries, -1
        )
        refreshed = []
        for camera_index, camera in enumerate(self.camera_keys):
            flat = pooled[:, :, camera_index].flatten(0, 1)
            if camera in raw_by_camera:
                indices, raw = raw_by_camera[camera]
                valid = self.vision_reattention[index // 2][camera](flat.index_select(0, indices), raw)
                flat = flat.index_copy(0, indices, valid)
            refreshed.append(flat.reshape(batch, image_steps, self.config.vision_pool_num_queries, -1))
        vision = torch.stack(refreshed, dim=2)
        return torch.cat((observation[:, :vision_start], vision.flatten(1, 3)), dim=1)

    def _encode_layers(
        self, observation: torch.Tensor, times: torch.Tensor, validity: torch.Tensor,
        *, vision_start: int, image_steps: int,
        raw_by_camera: dict[str, tuple[torch.Tensor, torch.Tensor]],
        last_step: int, cache: SPDObservationCache | None = None, rolling: bool = False,
    ) -> SPDObservationCache:
        layers = []
        for index, block in enumerate(self.observation_blocks):
            normalized = block.norm_attention(observation)
            key, value = block.attention.project_key_value(normalized, times)
            valid_keys = validity[:, None, :, None]
            key = key.masked_fill(~valid_keys, 0)
            value = value.masked_fill(~valid_keys, 0)
            key_times, key_validity = times, validity
            if cache is not None:
                previous = cache.layers[index]
                key = torch.cat((previous.key, key), dim=2)
                value = torch.cat((previous.value, value), dim=2)
                key_times = torch.cat((previous.times, times))
                key_validity = torch.cat((previous.validity, validity), dim=1)
            if rolling:
                keep = (last_step - key_times >= 0) & (last_step - key_times < self.config.attention_window_steps)
                key, value = key[:, :, keep], value[:, :, keep]
                key_times, key_validity = key_times[keep], key_validity[:, keep]
            layer = SPDLayerKV(key, value, key_times, key_validity)
            layers.append(layer)
            # The last level owns K/V only; there is no unused terminal block.
            if index < len(self.observation_blocks) - 1:
                mask = _causal_window_mask(times, key_times, self.config.attention_window_steps)
                mask = mask[None] | ~key_validity[:, None, :]
                observation = block.update(observation, normalized, times, layer, mask)
                if index % 2 == 1 and image_steps:
                    observation = self._refresh_vision(index, observation, vision_start, image_steps, raw_by_camera)
        return SPDObservationCache(tuple(layers), last_step)

    def encode_observations(self, batch: dict[str, Any]) -> SPDObservationCache:
        state = batch["state"]
        previous = batch["previous_actions"]
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 3
            or state.shape[1:] != (self.config.history_steps, self.config.state_dim)
            or state.shape[0] <= 0
        ):
            raise ValueError("state history shape does not match SPD contract")
        if (
            not isinstance(previous, torch.Tensor)
            or previous.ndim != 3
            or previous.shape != state.shape
        ):
            raise ValueError("previous_actions must match state history")
        if previous.device != state.device or previous.dtype != state.dtype:
            raise ValueError("state and previous_actions must share dtype and device")
        if not state.is_floating_point() or not previous.is_floating_point():
            raise ValueError("state and previous_actions must be floating-point tensors")
        if not torch.isfinite(state).all() or not torch.isfinite(previous).all():
            raise ValueError("state and previous_actions must contain finite values")
        if self.training:
            if self.config.observation_noise_std:
                state = state + torch.randn_like(state) * self.config.observation_noise_std
            if self.config.action_noise_std:
                previous = previous + torch.randn_like(previous) * self.config.action_noise_std
        state_tokens = self.state_embedding(state) + self.token_type.weight[0]
        previous_tokens = self.previous_actions_embedding(previous) + self.token_type.weight[1]
        state_tokens = torch.stack((state_tokens, previous_tokens), dim=2).flatten(1, 2)
        vision, raw_vision, camera_validity = self._vision_tokens(
            batch["images"], batch.get("camera_validity"),
            batch_size=state.shape[0], image_steps=self.image_steps, device=state.device,
        )
        observation = torch.cat((state_tokens, vision.flatten(1, 3)), dim=1)
        validity = torch.cat((
            torch.ones(state_tokens.shape[:2], dtype=torch.bool, device=state.device),
            camera_validity.repeat_interleave(self.config.vision_pool_num_queries, dim=2).flatten(1),
        ), dim=1)
        return self._encode_layers(
            observation, self.observation_times, validity,
            vision_start=state_tokens.shape[1], image_steps=self.image_steps,
            raw_by_camera=raw_vision, last_step=self.config.history_steps - 1,
        )

    @torch.no_grad()
    def append_observation(
        self,
        cache: SPDObservationCache | None,
        state: torch.Tensor,
        previous_actions: torch.Tensor,
        *,
        step: int,
        images: dict[str, torch.Tensor] | None = None,
        camera_validity: torch.Tensor | None = None,
    ) -> SPDObservationCache:
        """Append one 30 Hz observation tick to the deployment KV cache.

        ``images`` holds available ``[B,3,H,W]`` camera frames on image ticks.
        ``camera_validity`` is bool ``[B,3]`` in model camera order. Without it,
        all three images are required. No images means a state-only tick.
        """
        if (
            not isinstance(step, Integral)
            or isinstance(step, bool)
            or int(step) < 0
        ):
            raise ValueError("streaming step must be a non-negative integer")
        step = int(step)
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 2
            or state.shape[1] != self.config.state_dim
            or state.shape[0] <= 0
        ):
            raise ValueError("streaming state must have shape [B,54]")
        if not isinstance(previous_actions, torch.Tensor) or previous_actions.shape != state.shape:
            raise ValueError("streaming previous_actions must match state")
        if previous_actions.device != state.device or previous_actions.dtype != state.dtype:
            raise ValueError("state and previous_actions must share dtype and device")
        if not state.is_floating_point() or not previous_actions.is_floating_point():
            raise ValueError("streaming state and previous_actions must be floating-point tensors")
        if not torch.isfinite(state).all() or not torch.isfinite(previous_actions).all():
            raise ValueError("streaming state and previous_actions must contain finite values")
        if cache is not None:
            if not isinstance(cache, SPDObservationCache):
                raise ValueError("streaming cache must be an SPDObservationCache")
            if step <= cache.last_step:
                raise ValueError("streaming observation steps must strictly increase")
            if cache.layers[0].key.shape[0] != state.shape[0]:
                raise ValueError("streaming batch size cannot change within a cache")

        state = torch.stack(
            (
                self.state_embedding(state) + self.token_type.weight[0],
                self.previous_actions_embedding(previous_actions)
                + self.token_type.weight[1],
            ),
            dim=1,
        )
        token_times = torch.full((2,), step, device=state.device, dtype=torch.long)
        validity = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        raw_vision = {}
        image_steps = 0
        if images is not None:
            if not isinstance(images, dict):
                raise ValueError("streaming images must be a camera dictionary")
            prepared = {}
            for camera, image in images.items():
                if not isinstance(image, torch.Tensor) or image.ndim != 4:
                    raise ValueError(f"streaming {camera} image must have shape [B,3,H,W]")
                prepared[camera] = image[:, None]
            if camera_validity is not None:
                if not isinstance(camera_validity, torch.Tensor) or camera_validity.shape != (state.shape[0], len(self.camera_keys)):
                    raise ValueError("streaming camera_validity must have shape [B,3]")
                camera_validity = camera_validity[:, None]
            vision, raw_vision, camera_validity = self._vision_tokens(
                prepared, camera_validity, batch_size=state.shape[0], image_steps=1, device=state.device,
            )
            state = torch.cat((state, vision.flatten(1, 3)), dim=1)
            validity = torch.cat((validity, camera_validity.repeat_interleave(self.config.vision_pool_num_queries, dim=2).flatten(1)), dim=1)
            token_times = torch.full((state.shape[1],), step, device=state.device, dtype=torch.long)
            image_steps = 1
        elif camera_validity is not None:
            raise ValueError("camera_validity requires images (use an empty dict for entirely missing views)")
        return self._encode_layers(
            state, token_times, validity, vision_start=2, image_steps=image_steps,
            raw_by_camera=raw_vision, last_step=step, cache=cache, rolling=True,
        )

    def predict_cached_velocity(
        self, cache: SPDObservationCache, x_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one current action chunk from a rolling observation cache."""
        if not isinstance(cache, SPDObservationCache):
            raise ValueError("cache must be an SPDObservationCache")
        reference = cache.layers[0].key
        batch = reference.shape[0]
        expected = (batch, self.chunk_length, self.action_dim)
        if (
            not isinstance(x_t, torch.Tensor) or x_t.shape != expected
            or not x_t.is_floating_point() or not torch.isfinite(x_t).all()
            or x_t.device != reference.device
        ):
            raise ValueError(f"x_t must be finite floating {expected} on the cache device")
        if (
            not isinstance(t, torch.Tensor) or t.shape not in {(batch,), (batch, 1)}
            or t.device != reference.device or not t.is_floating_point()
            or not torch.isfinite(t).all() or bool(torch.any(t < 0)) or bool(torch.any(t > 1))
        ):
            raise ValueError("t must have one finite value in [0,1] per batch item")
        anchor = reference.new_tensor([cache.last_step], dtype=torch.long)
        return self._predict_chunks(cache, x_t[:, None], t.reshape(batch, 1), anchor)[:, 0]

    @torch.no_grad()
    def sample_actions_cached(
        self, cache: SPDObservationCache, num_steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate at the latest cache step, reusing all paired K/V levels."""
        if not isinstance(cache, SPDObservationCache):
            raise ValueError("cache must be an SPDObservationCache")
        anchors = cache.layers[0].times.new_tensor([cache.last_step])
        if noise is not None:
            expected = (cache.layers[0].key.shape[0], self.chunk_length, self.action_dim)
            if not isinstance(noise, torch.Tensor) or noise.shape != expected:
                raise ValueError(f"noise must have shape {expected}")
            noise = noise[:, None]
        return self._sample_chunks(cache, anchors, num_steps, noise)[:, 0]

    def _sample_chunks(
        self, conditioning: SPDObservationCache, anchors: torch.Tensor,
        num_steps: int, noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        reference = conditioning.layers[0].key
        shape = (reference.shape[0], anchors.numel(), self.chunk_length, self.action_dim)
        if noise is None:
            action = torch.randn(shape, device=reference.device, dtype=self.action_embedding.weight.dtype)
        else:
            if (not isinstance(noise, torch.Tensor) or noise.shape != shape
                or not noise.is_floating_point() or noise.device != reference.device
                or not torch.isfinite(noise).all()):
                raise ValueError(f"noise must be finite floating {shape} on the conditioning device")
            action = noise
        for step in range(num_steps):
            time = action.new_full(shape[:2], step / num_steps)
            action = action + self._predict_chunks(conditioning, action, time, anchors) / num_steps
        return action

    def predict_velocity(
        self, conditioning: SPDObservationCache, x_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict independently at each stride-aligned training anchor."""
        if not isinstance(conditioning, SPDObservationCache):
            raise ValueError("conditioning must come from encode_observations")
        reference = conditioning.layers[0].key
        batch = reference.shape[0]
        expected = (batch, self.image_steps, self.chunk_length, self.action_dim)
        if (
            not isinstance(x_t, torch.Tensor) or x_t.shape != expected
            or not x_t.is_floating_point() or not torch.isfinite(x_t).all()
            or x_t.device != reference.device
        ):
            raise ValueError(f"x_t must be finite floating {expected} on the conditioning device")
        if (
            not isinstance(t, torch.Tensor)
            or t.shape not in {(batch, self.image_steps), (batch, self.image_steps, 1, 1)}
            or t.device != reference.device or not t.is_floating_point()
            or not torch.isfinite(t).all() or bool(torch.any(t < 0)) or bool(torch.any(t > 1))
        ):
            raise ValueError("t must have one finite value in [0,1] per batch and chunk")
        anchors = self.action_times[::self.chunk_length]
        return self._predict_chunks(conditioning, x_t, t.reshape(batch, self.image_steps), anchors)

    def _predict_chunks(
        self, observation: SPDObservationCache, noised_action: torch.Tensor,
        time: torch.Tensor, anchors: torch.Tensor,
    ) -> torch.Tensor:
        """One paired-expert kernel for training anchors and latest-state inference."""
        action = self.action_embedding(noised_action)
        action = action + self.chunk_position_mlp(self.chunk_position)[None, None]
        action = action + self.flow_time(time)[:, :, None] + self.token_type.weight[3]
        action = action.flatten(1, 2)
        action_times = anchors.repeat_interleave(self.chunk_length)
        for block, layer in zip(self.action_expert, observation.layers, strict=True):
            context_mask = _action_context_mask(action_times, layer.times, self.config.attention_window_steps)
            invalid = torch.cat((
                ~layer.validity,
                torch.zeros((action.shape[0], action_times.numel()), device=action.device, dtype=torch.bool),
            ), dim=1)
            mask = context_mask[None] | invalid[:, None, :]
            action = block(action, layer, action_times, mask)
        return self.velocity_head(self.action_norm(action)).reshape_as(noised_action)

    def forward(
        self,
        batch: dict[str, Any],
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = batch["actions"]
        if (
            not isinstance(clean, torch.Tensor)
            or clean.ndim != 4
            or clean.shape[1:] != (self.image_steps, self.chunk_length, self.config.action_dim)
            or not clean.is_floating_point()
            or not torch.isfinite(clean).all()
        ):
            raise ValueError("actions must have shape [B,32,8,54] and finite values")
        batch_size = clean.shape[0]
        if noise is None:
            noise = torch.randn_like(clean)
        elif (
            not isinstance(noise, torch.Tensor)
            or noise.shape != clean.shape
            or not noise.is_floating_point()
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("noise must match actions and contain finite values")
        if t is None:
            t = torch.rand(batch_size, self.image_steps, device=clean.device)
        if (
            not isinstance(t, torch.Tensor)
            or t.shape != (batch_size, self.image_steps)
            or not t.is_floating_point()
            or not torch.isfinite(t).all()
            or bool(torch.any(t < 0.0))
            or bool(torch.any(t > 1.0))
        ):
            raise ValueError("t must have shape [B,32] and lie in [0,1]")
        expanded_time = t[:, :, None, None]
        point = (1 - expanded_time) * noise + expanded_time * clean
        target = clean - noise
        observation = self.encode_observations(batch)
        prediction = self.predict_velocity(observation, point, t)
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def sample_actions(
        self, batch: dict[str, Any], num_steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample using latest state 255, not last training anchor 248."""
        observation = self.encode_observations(batch)
        return self.sample_actions_cached(observation, num_steps, noise)

    @torch.no_grad()
    def sample_action_chunks(
        self, batch: dict[str, Any], num_steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aligned validation at anchors 0,8,...,248, with independent chunks."""
        observation = self.encode_observations(batch)
        return self._sample_chunks(
            observation, self.action_times[::self.chunk_length], num_steps, noise,
        )


def load_spd_checkpoint(
    model: SPDPolicy, checkpoint: Mapping, use_ema: bool = False,
) -> Mapping:
    """Load ABC full/slim SPD weights, optionally overlaying trainable EMA.

    Architecture metadata and every tensor are checked before mutating the model.
    Slim snapshots intentionally omit frozen DINO, which must be loaded separately.
    The operational DINO microbatch size is the only ignored config field.
    """
    if not isinstance(checkpoint, Mapping) or checkpoint.get("policy") != "spd":
        raise ValueError("checkpoint policy must be 'spd'")
    if checkpoint.get("architecture") != SPD_ARCHITECTURE:
        raise ValueError(f"checkpoint architecture must be {SPD_ARCHITECTURE!r}")
    saved = checkpoint.get("model_config")
    if not isinstance(saved, Mapping):
        raise ValueError("SPD checkpoint requires model_config metadata")
    expected_config = asdict(model.config)
    if set(saved) != set(expected_config):
        raise ValueError("SPD checkpoint model_config fields differ from SPDConfig")
    for name, expected in expected_config.items():
        actual = saved[name]
        if name == "camera_keys":
            if not isinstance(actual, (tuple, list)):
                raise ValueError("checkpoint camera_keys must be an ordered sequence")
            actual = tuple(actual)
            expected = tuple(expected)
        if name != "dino_frame_batch_size" and actual != expected:
            raise ValueError(f"SPD checkpoint model_config mismatch: {name}={actual!r}, expected {expected!r}")
    current = model.state_dict()
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("checkpoint contains no model weights")
    slim = {name: value for name, value in current.items() if not name.startswith("img_backbone.")}
    expected_state = current if any(str(name).startswith("img_backbone.") for name in model_state) else slim
    _validate_weights(model_state, expected_state, "SPD model")
    updates = dict(model_state)
    if use_ema:
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("model"), Mapping):
            raise ValueError("EMA requested but checkpoint has no EMA model")
        decay = ema.get("decay")
        if isinstance(decay, bool) or not isinstance(decay, Real) or not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("EMA decay must be finite and in [0,1]")
        trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
        _validate_weights(ema["model"], trainable, "SPD EMA")
        updates.update(ema["model"])
    current.update(updates)
    model.load_state_dict(current, strict=True)
    return checkpoint
