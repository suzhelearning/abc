import pytest

torch = pytest.importorskip("torch")

from spd_vr.config import SPDModelConfig
from spd_vr.policy import SPDPolicy, parameter_summary


class TinyVision(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = torch.nn.Linear(3, width)

    def encode_image_tokens(self, image):
        value = self.projection(image.mean(dim=(-2, -1)))
        return value[:, None, :].expand(-1, 5, -1)


def _batch():
    return {
        "qpos": torch.zeros(1, 256, 54),
        "previous_action": torch.zeros(1, 256, 54),
        "future_action": torch.ones(1, 32, 8, 54),
        "images": {
            name: torch.zeros(1, 32, 3, 16, 16)
            for name in ("top", "left_wrist", "right_wrist")
        },
    }


def test_spd_forward_backward_and_euler_sample():
    config = SPDModelConfig(
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        vision_queries=2,
        vit_embed_dim=32,
        vit_depth=1,
        vit_num_heads=4,
        vision_pool_num_heads=4,
        vision_pool_mlp_ratio=2,
    )
    model = SPDPolicy(config, vision_backbone=TinyVision(32))
    batch = _batch()
    loss = model(
        batch,
        noise=torch.zeros_like(batch["future_action"]),
        flow_time=torch.full((1, 32), 0.5),
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.velocity_head.weight.grad is not None
    summary = parameter_summary(model)
    assert summary["action_expert"] > 0

    model.eval()
    sampled = model.sample_actions(batch, num_steps=2)
    assert sampled.shape == (1, 8, 54)
    assert torch.isfinite(sampled).all()


def test_streaming_kv_matches_prefix_parallel_latest_chunk():
    config = SPDModelConfig(
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        vision_queries=2,
        vit_embed_dim=32,
        vit_depth=1,
        vit_num_heads=4,
        vision_pool_num_heads=4,
        vision_pool_mlp_ratio=2,
    )
    model = SPDPolicy(config, vision_backbone=TinyVision(32)).eval()
    batch = _batch()
    observation = model.encode_observations(batch)
    noised = torch.zeros_like(batch["future_action"])
    times = torch.full((1, 32), 0.5)
    full = model.predict_velocity(observation, noised, times)[:, -1]

    cache = None
    for step in range(249):
        images = None
        if step % 8 == 0:
            images = {
                name: batch["images"][name][:, step // 8]
                for name in model.camera_keys
            }
        cache = model.append_observation(
            cache,
            batch["qpos"][:, step],
            batch["previous_action"][:, step],
            step=step,
            images=images,
        )
    cached = model.predict_cached_velocity(cache, noised[:, -1], times[:, -1])
    assert torch.allclose(cached, full, atol=2e-5, rtol=2e-5)
    sampled = model.sample_actions_cached(cache, num_steps=2)
    assert sampled.shape == (1, 8, 54)


def test_load_dino_strips_nested_wrapper_prefixes(tmp_path):
    config = SPDModelConfig(
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        vision_queries=2,
        vit_embed_dim=32,
        vit_depth=1,
        vit_num_heads=4,
        vision_pool_num_heads=4,
        vision_pool_mlp_ratio=2,
    )
    model = SPDPolicy(config, vision_backbone=TinyVision(32))
    checkpoint = {
        "model": {
            f"module.backbone.dinov3_model.{name}": value.detach().clone()
            for name, value in model.img_backbone.state_dict().items()
        }
    }
    path = tmp_path / "dino_wrapped.pth"
    torch.save(checkpoint, path)
    missing, unexpected = model.load_dino(path)
    assert missing == []
    assert unexpected == []


@pytest.mark.parametrize("mutate", ["nan", "integer"])
def test_load_dino_rejects_nonfinite_or_nonfloating_parameters(tmp_path, mutate):
    config = SPDModelConfig(
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        vision_queries=2,
        vit_embed_dim=32,
        vit_depth=1,
        vit_num_heads=4,
        vision_pool_num_heads=4,
        vision_pool_mlp_ratio=2,
    )
    model = SPDPolicy(config, vision_backbone=TinyVision(32))
    state = {
        name: value.detach().clone()
        for name, value in model.img_backbone.state_dict().items()
    }
    if mutate == "nan":
        state["projection.weight"].fill_(float("nan"))
    else:
        state["projection.weight"] = state["projection.weight"].to(torch.int64)
    path = tmp_path / f"dino_{mutate}.pth"
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="invalid_values"):
        model.load_dino(path)
