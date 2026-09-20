"""Map official Hugging Face DINOv3 ViT weights to the native backbone."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import SPDConfig


def load_hf_dino_weights(path: Path, config: SPDConfig) -> dict[str, torch.Tensor]:
    """Require the native architecture's exact semantics, not just tensor shapes."""
    config_path = path.with_name("config.json")
    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"HF DINO weights require a valid adjacent {config_path}: {exc}") from exc
    expected = {
        "model_type": "dinov3_vit",
        "hidden_size": config.vit_embed_dim,
        "num_hidden_layers": config.vit_depth,
        "num_attention_heads": config.vit_num_heads,
        "intermediate_size": config.vit_embed_dim * 4,
        "num_channels": 3,
        "num_register_tokens": 4,
        "patch_size": 16,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "use_gated_mlp": False,
        "rope_theta": 100.0,
        "pos_embed_shift": None,
        "pos_embed_jitter": None,
        "pos_embed_rescale": 2.0,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    if not isinstance(metadata, dict):
        raise ValueError(f"{config_path}: expected a configuration object")
    mismatches = [key for key, value in expected.items() if key not in metadata or metadata[key] != value]
    if mismatches:
        raise ValueError(f"HF DINO configuration incompatible with native backbone: {mismatches}")
    source = load_file(str(path), device="cpu")

    def take(name: str) -> torch.Tensor:
        try:
            return source.pop(name)
        except KeyError as exc:
            raise ValueError(f"HF DINO checkpoint is missing {name}") from exc

    state = {
        "cls_token": take("embeddings.cls_token"),
        "storage_tokens": take("embeddings.register_tokens"),
        "mask_token": take("embeddings.mask_token").squeeze(1),
        "patch_embed.proj.weight": take("embeddings.patch_embeddings.weight"),
        "patch_embed.proj.bias": take("embeddings.patch_embeddings.bias"),
        "norm.weight": take("norm.weight"),
        "norm.bias": take("norm.bias"),
    }
    for index in range(config.vit_depth):
        src, dst = f"layer.{index}.", f"blocks.{index}."
        state[dst + "attn.qkv.weight"] = torch.cat([
            take(src + f"attention.{projection}_proj.weight") for projection in ("q", "k", "v")
        ])
        query_bias = take(src + "attention.q_proj.bias")
        # HF has no key bias. Native DINO represents this with a masked middle third.
        state[dst + "attn.qkv.bias"] = torch.cat([
            query_bias, torch.zeros_like(query_bias), take(src + "attention.v_proj.bias")
        ])
        state[dst + "attn.qkv.bias_mask"] = torch.cat([
            torch.ones_like(query_bias), torch.zeros_like(query_bias), torch.ones_like(query_bias)
        ])
        for native, hf in (
            ("norm1", "norm1"), ("norm2", "norm2"),
            ("attn.proj", "attention.o_proj"),
            ("mlp.fc1", "mlp.up_proj"), ("mlp.fc2", "mlp.down_proj"),
        ):
            for suffix in ("weight", "bias"):
                state[dst + native + "." + suffix] = take(src + hf + "." + suffix)
        state[dst + "ls1.gamma"] = take(src + "layer_scale1.lambda1")
        state[dst + "ls2.gamma"] = take(src + "layer_scale2.lambda1")
    if source:
        raise ValueError(f"HF DINO checkpoint has unexpected tensors: {sorted(source)}")
    # HF derives rotary frequencies at runtime rather than persisting this buffer.
    head_dim = config.vit_embed_dim // config.vit_num_heads
    state["rope_embed.periods"] = 100.0 ** (
        2 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2)
    )
    return state
