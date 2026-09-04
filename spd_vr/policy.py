"""Pure SPD flow-matching policy for 54-DoF simulated dexterity.

The module reuses ABC's released DINOv3 implementation while replacing the
single-frame language-conditioned DiT with the paper's long-horizon causal
observation trunk and a separately-parameterized action expert.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc_minimal.dit import AttentionPoolBlock, DinoVisionBackbone
from abc_minimal.flow import flow_interpolate

from .config import SPDModelConfig, validate_spd_model_config
from .contracts import ACTION_CHUNK, HISTORY_STEPS, IMAGE_STEPS, IMAGE_STRIDE


def _sincos(length: int, width: int) -> torch.Tensor:
    if width % 2:
        raise ValueError("sinusoidal embedding width must be even")
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        -math.log(10_000.0) * torch.arange(width // 2, dtype=torch.float32) / (width // 2)
    )
    return torch.cat((torch.sin(position * frequency), torch.cos(position * frequency)), dim=-1)


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


class TemporalAttention(nn.Module):
    """Multi-head attention with timestep RoPE on queries and keys."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("attention head width must be even")
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
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

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_times: torch.Tensor,
        key_times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, width = query.shape
        q = self.project_query(query, query_times)
        k, v = self.project_key_value(key_value, key_times)
        return self.attend_projected(q, k, v, mask)

    def project_query(
        self, query: torch.Tensor, query_times: torch.Tensor
    ) -> torch.Tensor:
        batch, query_length, _ = query.shape
        projected = self.query(query).reshape(
            batch, query_length, self.heads, self.head_width
        ).transpose(1, 2)
        return self._rope(projected, query_times)

    def project_key_value(
        self, key_value: torch.Tensor, key_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, key_length, _ = key_value.shape
        k = self.key(key_value).reshape(batch, key_length, self.heads, self.head_width).transpose(1, 2)
        v = self.value(key_value).reshape(batch, key_length, self.heads, self.head_width).transpose(1, 2)
        return self._rope(k, key_times), v

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, query_length, _ = query.shape
        width = self.heads * self.head_width
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=~mask[None, None]
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, width)
        return self.output(attended)


class ObservationBlock(nn.Module):
    def __init__(self, width: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = TemporalAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = FeedForward(width, ratio)

    def forward(
        self, value: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm_attention(value)
        attended = self.attention(normalized, normalized, times, times, mask)
        value = value + attended
        return value + self.mlp(self.norm_mlp(value))


class ActionExpertBlock(nn.Module):
    """Action queries attend causal observation and noised-action history."""

    def __init__(self, width: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(width)
        self.norm_key_value = nn.LayerNorm(width)
        self.attention = TemporalAttention(width, heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = FeedForward(width, ratio)

    def forward(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        action_times: torch.Tensor,
        key_times: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        key_value = torch.cat((observation, action), dim=1)
        attended = self.attention(
            self.norm_query(action),
            self.norm_key_value(key_value),
            action_times,
            key_times,
            mask,
        )
        action = action + attended
        return action + self.mlp(self.norm_mlp(action))


class VisionReattention(nn.Module):
    """Refresh pooled camera tokens from their original frozen DINO patch bank."""

    def __init__(self, hidden_size: int, vision_size: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, vision_size)
        self.attention = nn.MultiheadAttention(vision_size, heads, batch_first=True)
        self.output = nn.Linear(vision_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        pooled: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        # pooled [B, image_steps, cameras, queries, H]
        batch, steps, cameras, queries, hidden = pooled.shape
        flat = pooled.reshape(batch * steps * cameras, queries, hidden)
        query = self.query(self.norm(flat))
        refreshed, _ = self.attention(query, raw, raw, need_weights=False)
        return pooled + self.output(refreshed).reshape_as(pooled)


@dataclass(frozen=True, slots=True)
class SPDObservationCache:
    """Per-layer rolling observation KV plus final tokens for action queries."""

    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
    layer_times: tuple[torch.Tensor, ...]
    action_keys: tuple[torch.Tensor, ...]
    action_values: tuple[torch.Tensor, ...]
    observation: torch.Tensor
    observation_times: torch.Tensor
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

    def __init__(self, config: SPDModelConfig, vision_backbone: nn.Module | None = None) -> None:
        super().__init__()
        validate_spd_model_config(config)
        self.config = config
        hidden = config.hidden_size
        self.camera_keys = tuple(config.camera_keys)

        self.qpos_embedding = nn.Linear(config.state_dim, hidden)
        self.previous_action_embedding = nn.Linear(config.action_dim, hidden)
        self.action_embedding = nn.Linear(config.action_dim, hidden)
        self.token_type = nn.Embedding(4, hidden)
        self.register_buffer("chunk_position", _sincos(ACTION_CHUNK, hidden), persistent=True)
        self.chunk_position_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow_time = FlowTimeEmbedding(hidden)

        self.img_backbone = vision_backbone or DinoVisionBackbone(config)
        self.img_backbone.requires_grad_(False)
        self.vision_queries = nn.ParameterDict(
            {
                camera: nn.Parameter(torch.randn(1, config.vision_queries, config.vit_embed_dim) * 0.02)
                for camera in self.camera_keys
            }
        )
        # The paper's parameter count uses shared pooling weights with separate
        # learned queries for each camera.
        self.vision_pool = AttentionPoolBlock(
            config.vit_embed_dim,
            config.vision_pool_num_heads,
            config.vision_pool_mlp_ratio,
        )
        self.vision_projection = nn.Linear(config.vit_embed_dim, hidden)
        self.camera_embedding = nn.Embedding(len(self.camera_keys), hidden)

        self.observation_blocks = nn.ModuleList(
            ObservationBlock(hidden, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.vision_reattention = nn.ModuleList(
            VisionReattention(hidden, config.vit_embed_dim, config.vision_pool_num_heads)
            for _ in range(config.depth // 2)
        )
        self.action_expert = nn.ModuleList(
            ActionExpertBlock(hidden, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.action_norm = nn.LayerNorm(hidden)
        self.velocity_head = nn.Linear(hidden, config.action_dim)

        state_times = torch.arange(HISTORY_STEPS).repeat_interleave(2)
        image_times = torch.arange(0, HISTORY_STEPS, IMAGE_STRIDE).repeat_interleave(
            len(self.camera_keys) * config.vision_queries
        )
        observation_times = torch.cat((state_times, image_times))
        action_times = torch.arange(0, HISTORY_STEPS, IMAGE_STRIDE).repeat_interleave(ACTION_CHUNK)
        self.register_buffer("observation_times", observation_times, persistent=False)
        self.register_buffer("action_times", action_times, persistent=False)
        self.register_buffer(
            "observation_mask",
            _causal_window_mask(
                observation_times, observation_times, config.attention_window_steps
            ),
            persistent=False,
        )
        action_key_times = torch.cat((observation_times, action_times))
        self.register_buffer("action_key_times", action_key_times, persistent=False)
        self.register_buffer(
            "action_mask",
            _action_context_mask(
                action_times, observation_times, config.attention_window_steps
            ),
            persistent=False,
        )

    def set_dino_bfloat16(self, enabled: bool = True) -> None:
        setter = getattr(self.img_backbone, "set_bfloat16", None)
        if setter is not None:
            setter(enabled)

    def load_dino(self, checkpoint: str | Path) -> tuple[list[str], list[str]]:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
        required = dict(target.named_parameters())
        missing_parameters = sorted(set(required) - set(cleaned))
        shape_mismatch = sorted(
            name
            for name, parameter in required.items()
            if name in cleaned
            and (
                not isinstance(cleaned[name], torch.Tensor)
                or tuple(cleaned[name].shape) != tuple(parameter.shape)
            )
        )
        invalid_values = sorted(
            name
            for name, parameter in required.items()
            if name in cleaned
            and isinstance(cleaned[name], torch.Tensor)
            and (
                not cleaned[name].is_floating_point()
                or not torch.isfinite(cleaned[name]).all().item()
            )
        )
        if missing_parameters or shape_mismatch or invalid_values:
            raise ValueError(
                "DINOv3 checkpoint is incompatible: "
                f"missing={missing_parameters[:8]} "
                f"shape_mismatch={shape_mismatch[:8]} "
                f"invalid_values={invalid_values[:8]}"
            )
        missing, unexpected = target.load_state_dict(cleaned, strict=False)
        return list(missing), list(unexpected)

    def _vision_tokens(
        self,
        images: dict[str, torch.Tensor],
        *,
        expected_image_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(images, dict) or set(images) != set(self.camera_keys):
            raise ValueError(f"images must contain exactly {self.camera_keys}")
        pooled_by_camera, raw_by_camera = [], []
        image_steps = None
        batch_size = None
        for camera in self.camera_keys:
            image = images[camera]
            if not isinstance(image, torch.Tensor):
                raise ValueError(f"{camera} images must be torch tensors")
            if image.ndim != 5:
                raise ValueError(f"{camera} images must have shape [B,T,3,H,W]")
            if not image.is_floating_point():
                raise ValueError(f"{camera} images must be floating-point tensors")
            if image.shape[2] != 3 or image.shape[0] <= 0:
                raise ValueError(f"{camera} images must have a positive batch and three channels")
            if image_steps is None:
                image_steps = image.shape[1]
                if expected_image_steps is not None and image_steps != expected_image_steps:
                    raise ValueError(
                        f"images must contain {expected_image_steps} image steps, got {image_steps}"
                    )
                batch_size = image.shape[0]
            elif image.shape[1] != image_steps:
                raise ValueError("all camera streams must have the same length")
            if image.shape[0] != batch_size:
                raise ValueError("all camera streams must have the same batch size")
            if image.shape[3] <= 0 or image.shape[4] <= 0:
                raise ValueError(f"{camera} images must have positive spatial dimensions")
            if (image.is_floating_point() or image.is_complex()) and not torch.isfinite(image).all():
                raise ValueError(f"{camera} images must contain finite values")
            batch = image.shape[0]
            flat_image = image.reshape(batch * image_steps, *image.shape[2:])
            with torch.no_grad():
                raw = self.img_backbone.encode_image_tokens(flat_image)
            raw = raw.to(self.vision_queries[camera].dtype)
            query = self.vision_queries[camera].expand(raw.shape[0], -1, -1)
            pooled = self.vision_pool(raw, query)
            pooled_by_camera.append(
                pooled.reshape(batch, image_steps, self.config.vision_queries, -1)
            )
            raw_by_camera.append(raw.reshape(batch, image_steps, *raw.shape[1:]))

        pooled_vision = torch.stack(pooled_by_camera, dim=2)
        raw_vision = torch.stack(raw_by_camera, dim=2)
        projected = self.vision_projection(pooled_vision)
        camera_ids = torch.arange(len(self.camera_keys), device=projected.device)
        projected = projected + self.camera_embedding(camera_ids)[None, None, :, None, :]
        projected = projected + self.token_type.weight[2][None, None, None, None, :]
        return projected, raw_vision

    def encode_observations(self, batch: dict[str, Any]) -> torch.Tensor:
        qpos = batch["qpos"]
        previous = batch["previous_action"]
        if (
            not isinstance(qpos, torch.Tensor)
            or qpos.ndim != 3
            or qpos.shape[1:] != (HISTORY_STEPS, self.config.state_dim)
        ):
            raise ValueError("qpos history shape does not match SPD contract")
        if (
            not isinstance(previous, torch.Tensor)
            or previous.ndim != 3
            or previous.shape != qpos.shape
        ):
            raise ValueError("previous_action must match qpos history")
        if not qpos.is_floating_point() or not previous.is_floating_point():
            raise ValueError("qpos and previous_action must be floating-point tensors")
        if not torch.isfinite(qpos).all() or not torch.isfinite(previous).all():
            raise ValueError("qpos and previous_action must contain finite values")
        if self.training and self.config.observation_noise_std:
            qpos = qpos + torch.randn_like(qpos) * self.config.observation_noise_std
            previous = previous + torch.randn_like(previous) * self.config.action_noise_std
        qpos_tokens = self.qpos_embedding(qpos) + self.token_type.weight[0]
        previous_tokens = self.previous_action_embedding(previous) + self.token_type.weight[1]
        state_tokens = torch.stack((qpos_tokens, previous_tokens), dim=2).flatten(1, 2)
        vision, raw_vision = self._vision_tokens(
            batch["images"], expected_image_steps=IMAGE_STEPS
        )
        observation = torch.cat((state_tokens, vision.flatten(1, 3)), dim=1)

        vision_start = state_tokens.shape[1]
        for index, block in enumerate(self.observation_blocks):
            observation = block(
                observation, self.observation_times, self.observation_mask
            )
            if index % 2 == 1:
                pooled = observation[:, vision_start:].reshape_as(vision)
                raw = raw_vision.reshape(
                    -1, raw_vision.shape[-2], raw_vision.shape[-1]
                )
                pooled = self.vision_reattention[index // 2](pooled, raw)
                observation = torch.cat((observation[:, :vision_start], pooled.flatten(1, 3)), dim=1)
        return observation

    def append_observation(
        self,
        cache: SPDObservationCache | None,
        qpos: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        step: int,
        images: dict[str, torch.Tensor] | None = None,
    ) -> SPDObservationCache:
        """Append one 30 Hz observation tick to the deployment KV cache.

        ``images`` is supplied on image-subsample ticks and contains one
        ``[B,3,H,W]`` tensor for every camera. The caller owns chunk-boundary
        scheduling; normally it requests an action after every eight appends.
        """
        if (
            not isinstance(step, Integral)
            or isinstance(step, bool)
            or int(step) < 0
        ):
            raise ValueError("streaming step must be a non-negative integer")
        step = int(step)
        if (
            not isinstance(qpos, torch.Tensor)
            or qpos.ndim != 2
            or qpos.shape[1] != self.config.state_dim
            or qpos.shape[0] <= 0
        ):
            raise ValueError("streaming qpos must have shape [B,54]")
        if not isinstance(previous_action, torch.Tensor) or previous_action.shape != qpos.shape:
            raise ValueError("streaming previous_action must match qpos")
        if not qpos.is_floating_point() or not previous_action.is_floating_point():
            raise ValueError("streaming qpos and previous_action must be floating-point tensors")
        if not torch.isfinite(qpos).all() or not torch.isfinite(previous_action).all():
            raise ValueError("streaming qpos and previous_action must contain finite values")
        if cache is not None:
            if not isinstance(cache, SPDObservationCache):
                raise ValueError("streaming cache must be an SPDObservationCache")
            if step <= cache.last_step:
                raise ValueError("streaming observation steps must strictly increase")
            if cache.observation.shape[0] != qpos.shape[0]:
                raise ValueError("streaming batch size cannot change within a cache")

        state = torch.stack(
            (
                self.qpos_embedding(qpos) + self.token_type.weight[0],
                self.previous_action_embedding(previous_action)
                + self.token_type.weight[1],
            ),
            dim=1,
        )
        token_times = torch.full((2,), step, device=qpos.device, dtype=torch.long)
        vision = raw_vision = None
        if images is not None:
            if set(images) != set(self.camera_keys):
                raise ValueError(f"streaming images must contain {self.camera_keys}")
            prepared = {}
            for camera, image in images.items():
                if image.ndim != 4 or image.shape[0] != qpos.shape[0]:
                    raise ValueError(f"streaming {camera} image must have shape [B,3,H,W]")
                prepared[camera] = image[:, None]
            vision, raw_vision = self._vision_tokens(prepared, expected_image_steps=1)
            state = torch.cat((state, vision.flatten(1, 3)), dim=1)
            vision_times = torch.full(
                (len(self.camera_keys) * self.config.vision_queries,),
                step,
                device=qpos.device,
                dtype=torch.long,
            )
            token_times = torch.cat((token_times, vision_times))

        new_value = state
        next_keys, next_values, next_times = [], [], []
        for index, block in enumerate(self.observation_blocks):
            normalized = block.norm_attention(new_value)
            current_key, current_value = block.attention.project_key_value(
                normalized, token_times
            )
            if cache is None:
                key, value, key_times = current_key, current_value, token_times
            else:
                key = torch.cat((cache.layer_keys[index], current_key), dim=2)
                value = torch.cat((cache.layer_values[index], current_value), dim=2)
                key_times = torch.cat((cache.layer_times[index], token_times))
            keep = (step - key_times >= 0) & (
                step - key_times < self.config.attention_window_steps
            )
            key = key[:, :, keep]
            value = value[:, :, keep]
            key_times = key_times[keep]
            mask = _causal_window_mask(
                token_times, key_times, self.config.attention_window_steps
            )
            query = block.attention.project_query(normalized, token_times)
            attended = block.attention.attend_projected(query, key, value, mask)
            new_value = new_value + attended
            new_value = new_value + block.mlp(block.norm_mlp(new_value))
            if index % 2 == 1 and vision is not None and raw_vision is not None:
                pooled = new_value[:, 2:].reshape_as(vision)
                raw = raw_vision.reshape(
                    -1, raw_vision.shape[-2], raw_vision.shape[-1]
                )
                pooled = self.vision_reattention[index // 2](pooled, raw)
                new_value = torch.cat((new_value[:, :2], pooled.flatten(1, 3)), dim=1)
            next_keys.append(key)
            next_values.append(value)
            next_times.append(key_times)

        if cache is None:
            observation, observation_times = new_value, token_times
        else:
            observation = torch.cat((cache.observation, new_value), dim=1)
            observation_times = torch.cat((cache.observation_times, token_times))
        keep = (step - observation_times >= 0) & (
            step - observation_times < self.config.attention_window_steps
        )
        observation = observation[:, keep]
        observation_times = observation_times[keep]
        action_keys, action_values = [], []
        for index, block in enumerate(self.action_expert):
            normalized = block.norm_key_value(new_value)
            current_key, current_value = block.attention.project_key_value(
                normalized, token_times
            )
            if cache is None:
                key, value, times = current_key, current_value, token_times
            else:
                key = torch.cat((cache.action_keys[index], current_key), dim=2)
                value = torch.cat((cache.action_values[index], current_value), dim=2)
                times = torch.cat((cache.observation_times, token_times))
            action_keep = (step - times >= 0) & (
                step - times < self.config.attention_window_steps
            )
            action_keys.append(key[:, :, action_keep])
            action_values.append(value[:, :, action_keep])
        return SPDObservationCache(
            layer_keys=tuple(next_keys),
            layer_values=tuple(next_values),
            layer_times=tuple(next_times),
            action_keys=tuple(action_keys),
            action_values=tuple(action_values),
            observation=observation,
            observation_times=observation_times,
            last_step=int(step),
        )

    def predict_cached_velocity(
        self,
        cache: SPDObservationCache,
        noised_action: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one current 8-action chunk from a rolling observation cache."""
        if not isinstance(cache, SPDObservationCache):
            raise ValueError("cache must be an SPDObservationCache")
        batch = cache.observation.shape[0]
        expected = (batch, ACTION_CHUNK, self.config.action_dim)
        if (
            not isinstance(noised_action, torch.Tensor)
            or noised_action.shape != expected
            or not noised_action.is_floating_point()
            or not torch.isfinite(noised_action).all()
        ):
            raise ValueError(f"cached noised_action must have shape {expected}")
        if not isinstance(flow_time, torch.Tensor) or flow_time.shape not in {(batch,), (batch, 1)}:
            raise ValueError("cached flow_time must have one value per batch item")
        time = flow_time.reshape(batch)
        if (
            not time.is_floating_point()
            or not torch.isfinite(time).all()
            or bool(torch.any(time < 0.0))
            or bool(torch.any(time > 1.0))
        ):
            raise ValueError("cached flow_time must be finite and in [0,1]")
        action = self.action_embedding(noised_action)
        action = action + self.chunk_position_mlp(self.chunk_position)[None]
        action = action + self.flow_time(time)[:, None, :]
        action = action + self.token_type.weight[3]
        action_times = torch.full(
            (ACTION_CHUNK,),
            cache.last_step,
            device=action.device,
            dtype=torch.long,
        )
        mask = _action_context_mask(
            action_times,
            cache.observation_times,
            self.config.attention_window_steps,
        )
        for index, block in enumerate(self.action_expert):
            normalized_action = block.norm_query(action)
            action_key, action_value = block.attention.project_key_value(
                block.norm_key_value(action), action_times
            )
            key = torch.cat((cache.action_keys[index], action_key), dim=2)
            value = torch.cat((cache.action_values[index], action_value), dim=2)
            query = block.attention.project_query(normalized_action, action_times)
            attended = block.attention.attend_projected(query, key, value, mask)
            action = action + attended
            action = action + block.mlp(block.norm_mlp(action))
        return self.velocity_head(self.action_norm(action))

    @torch.no_grad()
    def sample_actions_cached(
        self,
        cache: SPDObservationCache,
        *,
        num_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate one deployment chunk while reusing the observation KV cache."""
        if not isinstance(cache, SPDObservationCache):
            raise ValueError("cache must be an SPDObservationCache")
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, Integral)
            or int(num_steps) <= 0
        ):
            raise ValueError("num_steps must be positive")
        num_steps = int(num_steps)
        batch = cache.observation.shape[0]
        action = torch.randn(
            (batch, ACTION_CHUNK, self.config.action_dim),
            device=cache.observation.device,
            dtype=cache.observation.dtype,
            generator=generator,
        )
        delta = 1.0 / num_steps
        for step in range(num_steps):
            time = torch.full(
                (batch,),
                step / num_steps,
                device=action.device,
                dtype=action.dtype,
            )
            action = action + delta * self.predict_cached_velocity(
                cache, action, time
            )
        return action

    def predict_velocity(
        self,
        observation: torch.Tensor,
        noised_action: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor) or observation.ndim != 3:
            raise ValueError("observation must have shape [B,observation_tokens,hidden]")
        if observation.shape[1] != self.observation_times.numel() or observation.shape[2] != self.config.hidden_size:
            raise ValueError("observation token shape does not match SPD contract")
        if not torch.isfinite(observation).all():
            raise ValueError("observation must contain finite values")
        if not isinstance(noised_action, torch.Tensor) or noised_action.ndim != 4:
            raise ValueError("noised_action must have shape [B,32,8,54]")
        batch = noised_action.shape[0]
        expected = (batch, IMAGE_STEPS, ACTION_CHUNK, self.config.action_dim)
        if noised_action.shape != expected or not noised_action.is_floating_point() or not torch.isfinite(noised_action).all():
            raise ValueError(f"noised_action must have shape {expected}")
        if observation.shape[0] != batch:
            raise ValueError("observation and noised_action batch sizes must match")
        if not isinstance(flow_time, torch.Tensor) or flow_time.shape not in {(batch, IMAGE_STEPS), (batch, IMAGE_STEPS, 1, 1)}:
            raise ValueError("flow_time must have one value per batch and action chunk")
        time = flow_time.reshape(batch, IMAGE_STEPS)
        if (
            not time.is_floating_point()
            or not torch.isfinite(time).all()
            or bool(torch.any(time < 0.0))
            or bool(torch.any(time > 1.0))
        ):
            raise ValueError("flow_time must be finite and in [0,1]")
        action = self.action_embedding(noised_action)
        action = action + self.chunk_position_mlp(self.chunk_position)[None, None, :, :]
        action = action + self.flow_time(time)[:, :, None, :]
        action = action + self.token_type.weight[3]
        action = action.flatten(1, 2)
        for block in self.action_expert:
            action = block(
                action,
                observation,
                self.action_times,
                self.action_key_times,
                self.action_mask,
            )
        velocity = self.velocity_head(self.action_norm(action))
        return velocity.reshape(batch, IMAGE_STEPS, ACTION_CHUNK, self.config.action_dim)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = batch["future_action"]
        if (
            not isinstance(clean, torch.Tensor)
            or clean.ndim != 4
            or clean.shape[1:] != (IMAGE_STEPS, ACTION_CHUNK, self.config.action_dim)
            or not clean.is_floating_point()
            or not torch.isfinite(clean).all()
        ):
            raise ValueError("future_action must have shape [B,32,8,54] and finite values")
        batch_size = clean.shape[0]
        if noise is None:
            noise = torch.randn_like(clean)
        elif (
            not isinstance(noise, torch.Tensor)
            or noise.shape != clean.shape
            or not noise.is_floating_point()
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("noise must match future_action and contain finite values")
        if flow_time is None:
            flow_time = torch.rand(batch_size, IMAGE_STEPS, device=clean.device)
        if (
            not isinstance(flow_time, torch.Tensor)
            or flow_time.shape != (batch_size, IMAGE_STEPS)
            or not flow_time.is_floating_point()
            or not torch.isfinite(flow_time).all()
            or bool(torch.any(flow_time < 0.0))
            or bool(torch.any(flow_time > 1.0))
        ):
            raise ValueError("flow_time must have shape [B,32] and lie in [0,1]")
        expanded_time = flow_time[:, :, None, None]
        point, target = flow_interpolate(clean, noise, expanded_time, data_at_one=True)
        observation = self.encode_observations(batch)
        prediction = self.predict_velocity(observation, point, flow_time)
        return F.mse_loss(prediction, target)

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Any],
        *,
        num_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, Integral)
            or int(num_steps) <= 0
        ):
            raise ValueError("num_steps must be positive")
        num_steps = int(num_steps)
        observation = self.encode_observations(batch)
        batch_size = batch["qpos"].shape[0]
        shape = (batch_size, IMAGE_STEPS, ACTION_CHUNK, self.config.action_dim)
        action = torch.randn(
            shape,
            device=batch["qpos"].device,
            dtype=batch["qpos"].dtype,
            generator=generator,
        )
        delta = 1.0 / num_steps
        for step in range(num_steps):
            time = torch.full(
                (batch_size, IMAGE_STEPS),
                step / num_steps,
                device=action.device,
                dtype=action.dtype,
            )
            action = action + delta * self.predict_velocity(observation, action, time)
        return action[:, -1]


def parameter_summary(model: SPDPolicy) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_vision": sum(
            parameter.numel() for parameter in model.img_backbone.parameters()
        ),
        "action_expert": sum(
            parameter.numel() for parameter in model.action_expert.parameters()
        ),
    }


def load_spd_checkpoint(
    model: SPDPolicy,
    path: str | Path,
    *,
    use_ema: bool = True,
) -> dict[str, Any]:
    """Load SPD trainable weights after DINO, rejecting foreign checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("not an SPD training checkpoint")
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("SPD checkpoint contains no model weights")
    ema_container = checkpoint.get("ema")
    ema_state = ema_container.get("model") if isinstance(ema_container, dict) else None
    source = ema_state if use_ema and isinstance(ema_state, dict) and ema_state else model_state
    if not isinstance(source, dict) or not source:
        raise ValueError("SPD checkpoint contains no model weights")
    current = model.state_dict()
    unexpected = sorted(set(source) - set(current))
    if unexpected:
        raise ValueError(f"checkpoint has incompatible keys: {unexpected[:8]}")
    shape_mismatch = [
        name
        for name, value in source.items()
        if not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(current[name].shape)
    ]
    if shape_mismatch:
        raise ValueError(f"checkpoint has incompatible tensor shapes: {shape_mismatch[:8]}")
    current.update(source)
    model.load_state_dict(current, strict=True)
    return checkpoint


__all__ = [
    "SPDObservationCache",
    "SPDPolicy",
    "load_spd_checkpoint",
    "parameter_summary",
]
