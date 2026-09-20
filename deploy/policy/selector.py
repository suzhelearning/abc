"""Detect whether a checkpoint is a DiT or VLA policy.

Current DiT and VLA checkpoints store weights under ``"model"``. The original
VLA release used ``"model_state_dict"``; both layouts are supported. Policy
kind comes from weight-key markers, not a layout-specific top-level key.
Reading with ``mmap=True`` avoids pulling the full multi-GB tensor payload into
memory.
"""

from pathlib import Path
from typing import Literal

import torch

from abc_minimal.checkpointing import model_state_dict

PolicyKind = Literal["dit", "vla", "spd"]

_VLA_MARKERS = ("vla.", "obs_pool.", "diffusion_head.")
_DIT_MARKERS = ("x_embedder", "pos_embed", "y_embedder")


def sniff_policy_kind(checkpoint_path: str) -> PolicyKind:
    ckpt = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if ckpt.get("architecture") == "spd-paired-kv-v2":
        raise ValueError("legacy SPD weights require scripts/convert_spd_checkpoint.py before ABC inference")
    if ckpt.get("policy") == "spd":
        from abc_minimal.spd import SPD_ARCHITECTURE

        if ckpt.get("architecture") != SPD_ARCHITECTURE:
            raise ValueError("unsupported SPD checkpoint architecture")
        return "spd"
    state = model_state_dict(ckpt)
    keys = list(state.keys())
    if any(k.startswith(_VLA_MARKERS) for k in keys):
        return "vla"
    if any(k.startswith(_DIT_MARKERS) for k in keys):
        return "dit"
    raise ValueError(
        f"could not classify checkpoint {checkpoint_path!r} as dit or vla "
        f"(sample keys: {keys[:6]})"
    )
