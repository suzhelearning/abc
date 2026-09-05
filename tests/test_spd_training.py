import pytest

torch = pytest.importorskip("torch")

from spd_vr.config import SPDModelConfig, SPDTrainConfig
from spd_vr.policy import SPDPolicy
from spd_vr.training import EMA, build_optimizers, optimization_step


class TinyVision(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = torch.nn.Linear(3, width)

    def encode_image_tokens(self, image):
        value = self.projection(image.mean(dim=(-2, -1)))
        return value[:, None, :].expand(-1, 5, -1)


def _config():
    return SPDModelConfig(
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


def test_shared_optimization_step_supports_tiny_overfit():
    torch.manual_seed(0)
    model_config = _config()
    model = SPDPolicy(model_config, vision_backbone=TinyVision(32))
    train_config = SPDTrainConfig(
        model=model_config,
        compile=False,
        dino_bf16=False,
        ema_half_life_steps=20.0,
    )
    muon, adamw = build_optimizers(model, train_config)
    ema = EMA(model, train_config.ema_half_life_steps)
    batch = _batch()
    losses = []
    for step in range(12):
        torch.manual_seed(100 + step)
        loss, grad_norm = optimization_step(
            model,
            muon,
            adamw,
            ema,
            batch,
            max_grad_norm=train_config.optim.max_grad_norm,
        )
        losses.append(float(loss))
        assert torch.isfinite(grad_norm)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(value).all() for value in ema.state.values())


def test_optimization_step_rejects_invalid_gradient_budget():
    model_config = _config()
    model = SPDPolicy(model_config, vision_backbone=TinyVision(32))
    train_config = SPDTrainConfig(model=model_config, compile=False, dino_bf16=False)
    muon, adamw = build_optimizers(model, train_config)
    ema = EMA(model, train_config.ema_half_life_steps)
    with pytest.raises(ValueError, match="max_grad_norm"):
        optimization_step(
            model,
            muon,
            adamw,
            ema,
            _batch(),
            max_grad_norm=0,
        )
