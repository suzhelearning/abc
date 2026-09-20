"""Behavioral regressions for paired SPD conditioning and deployment caching."""

from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from abc_minimal.config import SPDConfig
from abc_minimal.spd import (
    SPD_ARCHITECTURE,
    SPDLayerKV,
    SPDObservationCache,
    SPDPolicy,
    load_spd_checkpoint,
    validate_spd_config,
)


class TinyVision(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(3, width)
        self.frames_seen = 0

    def encode_image_tokens(self, images):
        self.frames_seen += images.shape[0]
        patches = images.flatten(2).transpose(1, 2)
        return self.projection(patches[:, ::16])


def make_policy(*, depth=3, history_steps=256):
    torch.manual_seed(71)
    config = SPDConfig(
        hidden_size=16, depth=depth, num_heads=2, mlp_ratio=2,
        history_steps=history_steps, vit_embed_dim=16, vit_depth=1,
        vit_num_heads=2, vision_pool_num_heads=2, vision_pool_mlp_ratio=2,
        vision_pool_num_queries=2, dino_frame_batch_size=3,
        observation_noise_std=0, action_noise_std=0,
    )
    return SPDPolicy(config, TinyVision(config.vit_embed_dim)).eval()


def make_batch(model, batch_size=2, *, images=True):
    config = model.config
    result = {
        "state": torch.randn(batch_size, config.history_steps, 54),
        "previous_actions": torch.randn(batch_size, config.history_steps, 54),
        "actions": torch.randn(batch_size, model.image_steps, 8, 54),
        "images": {},
        "camera_validity": torch.zeros(batch_size, model.image_steps, 3, dtype=torch.bool),
    }
    if images:
        result["images"] = {
            camera: torch.randn(batch_size, model.image_steps, 3, 8, 8)
            for camera in model.camera_keys
        }
        result["camera_validity"][:, :, 0] = True
        result["camera_validity"][0, ::2, 1] = True
        if batch_size > 1:
            result["camera_validity"][1, :, 2] = True
    return result


def snapshot(model):
    return {
        "policy": "spd", "architecture": SPD_ARCHITECTURE,
        "model_config": asdict(model.config),
        "model": {
            name: value.detach().clone() for name, value in model.state_dict().items()
            if not name.startswith("img_backbone.")
        },
    }


@torch.no_grad()
def test_full_and_rolling_cache_agree_at_every_anchor_and_latest_255():
    model = make_policy()
    batch = make_batch(model)
    full = model.encode_observations(batch)
    noise = torch.randn_like(batch["actions"])
    time = torch.rand(2, model.image_steps)
    full_velocity = model.predict_velocity(full, noise, time)
    cache = None
    for step in range(256):
        kwargs = {}
        if step % 8 == 0:
            index = step // 8
            kwargs = {
                "images": {name: frames[:, index] for name, frames in batch["images"].items()},
                "camera_validity": batch["camera_validity"][:, index],
            }
        cache = model.append_observation(
            cache, batch["state"][:, step], batch["previous_actions"][:, step],
            step=step, **kwargs,
        )
        for layer in cache.layers:
            assert (layer.times <= step).all()
            assert (layer.times > step - 32).all()
            assert layer.times.numel() <= 32 * 2 + 4 * 3 * model.config.vision_pool_num_queries
        if step % 8 == 0:
            cached_velocity = model.predict_cached_velocity(cache, noise[:, index], time[:, index])
            torch.testing.assert_close(cached_velocity, full_velocity[:, index], atol=3e-6, rtol=3e-5)
    latest_noise = torch.randn(2, 8, 54)
    latest = model.sample_actions(batch, num_steps=2, noise=latest_noise)
    torch.testing.assert_close(
        latest, model.sample_actions_cached(cache, num_steps=2, noise=latest_noise),
        atol=3e-6, rtol=3e-5,
    )
    changed = {**batch, "state": batch["state"].clone()}
    changed["state"][:, 255] += torch.randn(2, 54) * 4
    assert not torch.allclose(latest, model.sample_actions(changed, num_steps=2, noise=latest_noise))
    # The final seven states are future information for every training anchor.
    torch.testing.assert_close(
        model.predict_velocity(model.encode_observations(changed), noise, time), full_velocity,
    )
    expected = noise + model.predict_velocity(full, noise, torch.zeros_like(time)) / 2
    expected = expected + model.predict_velocity(full, expected, torch.full_like(time, 0.5)) / 2
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=2, noise=noise), expected)


@torch.no_grad()
def test_paired_normalized_layer_input_kv_and_direct_action_consumption():
    model = make_policy(depth=3, history_steps=16)
    batch = make_batch(model, images=False)
    projected = []
    hooks = []
    # The paired normalized-input projection is an architectural invariant:
    # catching post-block K/V would otherwise preserve shapes and load silently.
    for block in model.observation_blocks:
        def capture(module, args, output, attention=block.attention):
            projected.append(attention.project_key_value(output, model.observation_times))
        hooks.append(block.norm_attention.register_forward_hook(capture))
    try:
        conditioning = model.encode_observations(batch)
    finally:
        for hook in hooks:
            hook.remove()
    for layer, (key, value) in zip(conditioning.layers, projected, strict=True):
        valid = layer.validity[:, None, :, None]
        torch.testing.assert_close(layer.key, key.masked_fill(~valid, 0))
        torch.testing.assert_close(layer.value, value.masked_fill(~valid, 0))
    x_t, t = torch.randn_like(batch["actions"]), torch.rand(2, model.image_steps)
    original = model.predict_velocity(conditioning, x_t, t)
    # Each same-layer observation bank must influence the action expert directly.
    for index, layer in enumerate(conditioning.layers):
        values = layer.value + torch.randn_like(layer.value) * layer.validity[:, None, :, None] * 5
        layers = list(conditioning.layers)
        layers[index] = SPDLayerKV(layer.key, values, layer.times, layer.validity)
        changed = model.predict_velocity(SPDObservationCache(tuple(layers), conditioning.last_step), x_t, t)
        assert not torch.allclose(original, changed)


@torch.no_grad()
def test_action_chunks_do_not_attend_other_chunks_or_future_observations():
    model = make_policy(history_steps=32)
    batch = make_batch(model, images=False)
    noise, time = torch.randn_like(batch["actions"]), torch.rand(2, model.image_steps)
    original = model.predict_velocity(model.encode_observations(batch), noise, time)
    changed_noise = noise.clone()
    changed_noise[:, 1] *= 20
    changed = model.predict_velocity(model.encode_observations(batch), changed_noise, time)
    torch.testing.assert_close(changed[:, [0, 2, 3]], original[:, [0, 2, 3]])
    batch["state"][:, 1:] *= 10
    batch["previous_actions"][:, 1:] *= 10
    changed = model.predict_velocity(model.encode_observations(batch), noise, time)
    torch.testing.assert_close(changed[:, 0], original[:, 0])


@torch.no_grad()
def test_masked_pixels_are_unread_and_mixed_validity_is_per_example():
    model = make_policy(history_steps=16)
    batch = make_batch(model)
    noise = torch.randn_like(batch["actions"])
    expected = model.sample_action_chunks(batch, num_steps=1, noise=noise)
    assert model.img_backbone.frames_seen == int(batch["camera_validity"].sum())
    model.img_backbone.frames_seen = 0
    for index, name in enumerate(model.camera_keys):
        batch["images"][name][~batch["camera_validity"][:, :, index]] = float("nan")
    actual = model.sample_action_chunks(batch, num_steps=1, noise=noise)
    torch.testing.assert_close(actual, expected)
    assert model.img_backbone.frames_seen == int(batch["camera_validity"].sum())
    batch["images"]["right_wrist"][1] *= 10
    changed = model.sample_action_chunks(batch, num_steps=1, noise=noise)
    torch.testing.assert_close(changed[0], actual[0])
    assert not torch.allclose(changed[1], actual[1])
    batch["camera_validity"].zero_()
    batch["images"] = {}
    model.img_backbone.frames_seen = 0
    missing = model.sample_action_chunks(batch, num_steps=1, noise=noise)
    assert model.img_backbone.frames_seen == 0
    batch["images"] = {name: torch.full((2, model.image_steps, 3, 8, 8), float("nan")) for name in model.camera_keys}
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=1, noise=noise), missing)
    assert model.img_backbone.frames_seen == 0
    batch["camera_validity"][0, 0, 2] = True
    batch["images"].pop("right_wrist")
    with pytest.raises(ValueError, match="missing"):
        model.encode_observations(batch)


def test_right_camera_branch_trains_while_dino_stays_frozen_eval():
    model = make_policy(depth=8, history_steps=16).train()
    assert not model.img_backbone.training
    assert not any(module.training for module in model.img_backbone.modules())
    batch = make_batch(model)
    loss = model(batch, noise=torch.randn_like(batch["actions"]), t=torch.full((2, 2), 0.4))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.img_backbone.parameters())
    query_grad = model.vision_queries["right_wrist"].grad
    assert query_grad is not None and query_grad.abs().sum() > 0
    for refresh in model.vision_reattention:
        grad = refresh["right_wrist"].attention.out_proj.weight.grad
        assert grad is not None and grad.abs().sum() > 0
    for level in model.observation_blocks:
        grad = level.attention.value.weight.grad
        assert grad is not None and grad.abs().sum() > 0


@torch.no_grad()
def test_noise_to_data_flow_direction_and_positive_euler_integration():
    model = make_policy(depth=1, history_steps=16)
    model.velocity_head.weight.zero_()
    model.velocity_head.bias.fill_(2)
    batch = make_batch(model, images=False)
    batch["actions"].fill_(3)
    noise = torch.ones_like(batch["actions"])
    assert model(batch, noise=noise, t=torch.full((2, 2), 0.37)).item() == 0
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=10, noise=noise), batch["actions"])
    torch.testing.assert_close(
        model.sample_actions(batch, num_steps=10, noise=noise[:, 0]), batch["actions"][:, 0],
    )


@pytest.mark.parametrize("architecture", ["spd-paired-kv-v2", "dit", "vla"])
def test_rejects_foreign_architectures(architecture):
    model = make_policy(history_steps=16)
    checkpoint = snapshot(model)
    checkpoint["architecture"] = architecture
    with pytest.raises(ValueError, match="architecture"):
        load_spd_checkpoint(model, checkpoint)


@pytest.mark.parametrize("field,value", [
    ("attention_window_steps", 16), ("image_stride", 4),
    ("camera_keys", ["right_wrist", "left_wrist", "top"]),
    ("observation_noise_std", 0.5),
])
def test_rejects_shape_compatible_semantic_mismatch(field, value):
    model = make_policy(history_steps=16)
    checkpoint = snapshot(model)
    checkpoint["model_config"][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        load_spd_checkpoint(model, checkpoint)


@torch.no_grad()
def test_slim_checkpoint_and_ema_select_the_requested_policy():
    model = make_policy(depth=1, history_steps=16)
    model.velocity_head.weight.zero_()
    model.velocity_head.bias.fill_(1)
    checkpoint = snapshot(model)
    batch = make_batch(model, images=False)
    noise = torch.zeros_like(batch["actions"])
    with pytest.raises(ValueError, match="EMA requested"):
        load_spd_checkpoint(model, checkpoint, use_ema=True)
    checkpoint["ema"] = {
        "decay": 0.99,
        "model": {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad},
    }
    checkpoint["ema"]["model"]["velocity_head.bias"].fill_(3)
    load_spd_checkpoint(model, checkpoint, use_ema=True)
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=1, noise=noise), torch.full_like(noise, 3))
    load_spd_checkpoint(model, checkpoint)
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=1, noise=noise), torch.ones_like(noise))
    checkpoint["model"]["velocity_head.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        load_spd_checkpoint(model, checkpoint, use_ema=True)
    torch.testing.assert_close(model.sample_action_chunks(batch, num_steps=1, noise=noise), torch.ones_like(noise))


def test_checkpoint_requires_complete_model_and_ema_keys():
    model = make_policy(depth=1, history_steps=16)
    checkpoint = snapshot(model)
    checkpoint["model"].pop("flow_time.frequencies")
    with pytest.raises(ValueError, match="missing"):
        load_spd_checkpoint(model, checkpoint)
    checkpoint = snapshot(model)
    checkpoint["ema"] = {"decay": 0.99, "model": {}}
    with pytest.raises(ValueError, match="missing"):
        load_spd_checkpoint(model, checkpoint, use_ema=True)


def test_tianji_joint_dimensions_are_not_generalized():
    config = SPDConfig()
    assert validate_spd_config(config) == []
    assert validate_spd_config(replace(config, state_dim=56, action_dim=56))
