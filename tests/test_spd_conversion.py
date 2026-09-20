"""Migration math and fail-closed artifact contracts, without legacy imports."""

from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest
import torch
from torch import nn

from abc_minimal.config import SPDConfig
from abc_minimal.preprocess import normalize, parse_norm_stats, unnormalize
from abc_minimal.spd import ObservationBlock, SPDPolicy, VisionReattention, load_spd_checkpoint
from abc_minimal.spd_conversion import (
    OFFICIAL_DINO_SHA256,
    SOURCE_ARCHITECTURE,
    _atomic_save,
    _convert_normalization,
    _convert_payload,
    _convert_weights,
    _fold_reattention,
    _source_config,
    convert_spd_checkpoint,
)


class OldReattention(nn.Module):
    """Locally implemented source math, not a cross-repository model adapter."""

    def __init__(self, width, heads):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, pooled, raw):
        refreshed, _ = self.attention(self.query(self.norm(pooled)), raw, raw, need_weights=False)
        return pooled + self.output(refreshed)


def old_config(config):
    saved = asdict(config)
    saved["vision_queries"] = saved.pop("vision_pool_num_queries")
    return saved


def source_normalization():
    epsilon = np.float32(1e-6)
    scales = np.resize(np.array([
        epsilon / 8, epsilon, np.nextafter(epsilon, np.float32(np.inf)),
        0.0001, 0.0325, 1.37, 300.0, 1e12, np.finfo(np.float32).max,
    ], dtype=np.float32), 54)
    means = np.linspace(-0.7, 0.9, 54, dtype=np.float32)
    return {"qpos_mean": means.tolist(), "qpos_std": scales.tolist(),
            "action_mean": (-means).tolist(), "action_std": scales[::-1].tolist()}


@pytest.fixture
def migration_fixture():
    torch.manual_seed(53)
    config = SPDConfig(
        hidden_size=16, depth=8, num_heads=2, mlp_ratio=2,
        history_steps=16, vit_embed_dim=16, vit_depth=1,
        vit_num_heads=2, vision_pool_num_heads=2, vision_pool_mlp_ratio=2,
        vision_pool_num_queries=2, dino_frame_batch_size=3,
    )
    target = SPDPolicy(config).eval()
    reference = deepcopy(target)
    # Explicit old graph. Its terminal block and fourth refresh exist but are
    # unconsumed by the paired-K/V observation loop shared by both graphs.
    reference.observation_blocks[-1] = ObservationBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
    reference.vision_reattention = nn.ModuleList([
        nn.ModuleDict({camera: OldReattention(config.hidden_size, config.vision_pool_num_heads)
                       for camera in config.camera_keys})
        for _ in range(config.depth // 2)
    ])
    with torch.no_grad():
        # Persisted buffers must survive migration rather than being regenerated.
        reference.chunk_position.copy_(torch.randn_like(reference.chunk_position))
        reference.flow_time.frequencies.copy_(torch.randn_like(reference.flow_time.frequencies) * 2)
        for block in reference.vision_reattention:
            for module in block.values():
                module.norm.weight.uniform_(0.4, 1.7)
                module.norm.bias.uniform_(-0.3, 0.6)
    reference.eval()

    def rename(name):
        return name.replace("state_embedding.", "qpos_embedding.").replace(
            "previous_actions_embedding.", "previous_action_embedding.")

    raw = {rename(name): value.detach().clone() for name, value in reference.state_dict().items()
           if not name.startswith("img_backbone.")}
    averaged = {rename(name): value.detach().clone() * 0.91 + 0.017
                for name, value in reference.named_parameters() if value.requires_grad}
    source = {
        "architecture": SOURCE_ARCHITECTURE, "config": {"model": old_config(config)},
        "model": raw, "ema": {"model": averaged, "decay": math.exp(math.log(0.5) / 20)},
        "normalization": source_normalization(), "step": 10000,
        "dino_checkpoint_sha256": OFFICIAL_DINO_SHA256,
        "dataset_contract": {
            "config": {"camera_names": ["top", "left_wrist"]},
            "model_camera_keys": list(config.camera_keys),
            "observed_camera_keys": ["top", "left_wrist"],
            "unavailable_camera_keys": ["right_wrist"],
        },
        "muon": {"unusable": torch.ones(1)}, "adamw": {"unusable": torch.ones(1)},
        "rng": {"unusable": torch.ones(1)}, "rng_by_rank": [{"unusable": torch.ones(1)}],
    }
    return target, reference, source


def test_reattention_fold_preserves_normalized_cross_attention_and_input_gradients():
    torch.manual_seed(39)
    old = OldReattention(24, 3).eval()
    with torch.no_grad():
        old.norm.weight.uniform_(0.2, 1.8)
        old.norm.bias.uniform_(-0.7, 0.4)
    new = VisionReattention(24, 24, 3).eval()
    new.load_state_dict(_fold_reattention(old.state_dict()), strict=True)
    pooled = (torch.randn(3, 5, 24) * 2.7 + 0.6).requires_grad_()
    raw = (torch.randn(3, 11, 24) * 1.9 - 0.8).requires_grad_()
    expected = old(pooled, raw)
    actual = new(pooled, raw)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(expected)
    old_grads = torch.autograd.grad((expected * probe).sum(), (pooled, raw))
    new_grads = torch.autograd.grad((actual * probe).sum(), (pooled, raw))
    for actual_grad, expected_grad in zip(new_grads, old_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)


@torch.no_grad()
@pytest.mark.parametrize("use_ema", [False, True])
def test_converted_full_small_model_preserves_velocity_and_buffers(migration_fixture, use_ema):
    target, reference, source = migration_fixture
    checkpoint = _convert_payload(source, target, dino_sha256=OFFICIAL_DINO_SHA256,
                                  provenance={"path": "fixture.pt", "sha256": "fixture"})
    load_spd_checkpoint(target, checkpoint, use_ema=use_ema)
    if use_ema:
        parameters = dict(reference.named_parameters())
        for name, value in source["ema"]["model"].items():
            name = name.replace("qpos_embedding.", "state_embedding.").replace(
                "previous_action_embedding.", "previous_actions_embedding.")
            parameters[name].copy_(value)
    batch = {
        "state": torch.randn(1, 16, 54), "previous_actions": torch.randn(1, 16, 54),
        "images": {camera: torch.randn(1, 2, 3, 32, 32) for camera in target.camera_keys},
        "camera_validity": torch.tensor([[[True, False, False], [True, True, False]]]),
    }
    noise = torch.randn(1, 2, 8, 54)
    flow_time = torch.tensor([[0.13, 0.82]])
    expected = reference.predict_velocity(reference.encode_observations(batch), noise, flow_time)
    actual = target.predict_velocity(target.encode_observations(batch), noise, flow_time)
    torch.testing.assert_close(actual, expected, atol=8e-6, rtol=8e-5)
    torch.testing.assert_close(target.chunk_position, reference.chunk_position, rtol=0, atol=0)
    torch.testing.assert_close(target.flow_time.frequencies, reference.flow_time.frequencies, rtol=0, atol=0)
    assert checkpoint["global_step"] == 10000
    assert checkpoint["weights_only"] is True and checkpoint["exact_resume"] is False
    assert checkpoint["source_camera_keys"] == ["top", "left_wrist"]
    assert not {"optimizer", "scheduler", "muon", "adamw", "rng", "rng_by_rank", "dataset_contract"} & checkpoint.keys()


def test_normalization_preserves_physical_inputs_and_actions():
    source = source_normalization()
    converted = parse_norm_stats(_convert_normalization(source))
    latent = np.linspace(-0.7, 0.8, 54, dtype=np.float32)
    for old, new in (("qpos", "state"), ("action", "actions")):
        mean = np.asarray(source[old + "_mean"], dtype=np.float32)
        effective = np.maximum(np.asarray(source[old + "_std"], dtype=np.float32), np.float32(1e-6))
        physical = latent * effective + mean
        expected_normalized = (physical - mean) / effective
        np.testing.assert_allclose(normalize(physical, converted[new]), expected_normalized, rtol=2e-7, atol=1e-7)
        np.testing.assert_allclose(unnormalize(latent, converted[new]), physical, rtol=2e-7, atol=1e-7)
        reconstructed = converted[new]["std"] + np.float32(1e-6)
        np.testing.assert_array_max_ulp(reconstructed, effective, maxulp=1)
        assert np.all(converted[new]["std"] > 0)
    assert converted["state"]["std"][0] == np.nextafter(np.float32(0), np.float32(1))


@pytest.mark.parametrize("which", ["model", "ema"])
@pytest.mark.parametrize("damage", ["extra", "missing_dead", "nonfinite_dead", "wrong_shape", "wrong_dtype"])
def test_source_tensor_graph_is_fail_closed(migration_fixture, which, damage):
    target, _, source = migration_fixture
    state = source["model"] if which == "model" else source["ema"]["model"]
    if damage == "extra":
        state["unrecognized.weight"] = torch.ones(1)
    elif damage == "missing_dead":
        del state["vision_reattention.3.top.query.weight"]
    elif damage == "nonfinite_dead":
        state["observation_blocks.7.mlp.net.0.weight"][0, 0] = float("nan")
    elif damage == "wrong_shape":
        state["qpos_embedding.weight"] = torch.ones(16, 53)
    else:
        state["qpos_embedding.weight"] = state["qpos_embedding.weight"].half()
    with pytest.raises(ValueError):
        _convert_payload(source, target, dino_sha256=OFFICIAL_DINO_SHA256, provenance={})


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 384), ("state_dim", 56), ("history_steps", 128),
    ("image_stride", 4), ("chunk_length", 16), ("depth", 6),
    ("vision_queries", 5), ("observation_noise_std", float("nan")),
    ("new_unknown_field", 1),
])
def test_production_source_config_rejects_incompatible_graph(field, value):
    saved = old_config(SPDConfig())
    saved[field] = value
    with pytest.raises(ValueError):
        _source_config({"architecture": SOURCE_ARCHITECTURE, "config": {"model": saved}})


def test_old_architecture_and_dino_identity_are_required(migration_fixture, tmp_path):
    target, _, source = migration_fixture
    with pytest.raises(ValueError, match="architecture"):
        _source_config({**source, "architecture": "spd-paired-kv-v1"})
    with pytest.raises(ValueError, match="DINO hash"):
        _convert_payload(source, target, dino_sha256="0" * 64, provenance={})
    source_path, dino_path = tmp_path / "source.pt", tmp_path / "model.safetensors"
    source_path.write_bytes(b"not deserialized before DINO validation")
    dino_path.write_bytes(b"not the official weights")
    output = tmp_path / "output.pt"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        convert_spd_checkpoint(source_path, output, dino_path)
    assert not output.exists()


def test_unequal_attention_widths_cannot_be_folded(migration_fixture):
    target, _, source = migration_fixture
    incompatible = SPDPolicy(replace(target.config, hidden_size=32))
    with pytest.raises(ValueError, match="unequal hidden/vision widths"):
        _convert_weights(source["model"], incompatible)


@pytest.mark.parametrize("damage", ["negative_std", "nonfinite_mean", "missing", "camera_mismatch", "ema_decay"])
def test_malformed_metadata_is_rejected(migration_fixture, damage):
    target, _, source = migration_fixture
    if damage == "negative_std":
        source["normalization"]["qpos_std"][0] = -1
    elif damage == "nonfinite_mean":
        source["normalization"]["action_mean"][0] = float("inf")
    elif damage == "missing":
        del source["normalization"]["action_std"]
    elif damage == "camera_mismatch":
        source["dataset_contract"]["observed_camera_keys"] = ["top", "right_wrist"]
    else:
        source["ema"]["decay"] = 1.01
    with pytest.raises(ValueError):
        _convert_payload(source, target, dino_sha256=OFFICIAL_DINO_SHA256, provenance={})


def test_atomic_save_refuses_overwrite_and_cleans_temporary_files(tmp_path, monkeypatch):
    output = tmp_path / "converted.pt"
    _atomic_save({"model": {"weight": torch.tensor([1.5])}}, output)
    with pytest.raises(FileExistsError):
        _atomic_save({"model": {"weight": torch.tensor([9.0])}}, output)
    torch.testing.assert_close(torch.load(output, weights_only=True)["model"]["weight"], torch.tensor([1.5]))

    def fail_save(*args, **kwargs):
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    incomplete = tmp_path / "incomplete.pt"
    with pytest.raises(OSError, match="interrupted"):
        _atomic_save({}, incomplete)
    assert not incomplete.exists()
    assert set(tmp_path.iterdir()) == {output}
