from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from abc_minimal.config import SPDConfig
from abc_minimal.policy import SPDInferencePolicy, SPDPolicyConfig
from abc_minimal.preprocess import normalize, resize_pad_normalize_batch, unnormalize
from abc_minimal.spd import SPDPolicy, SPD_ARCHITECTURE


@pytest.fixture
def inference(tmp_path):
    torch.manual_seed(23)
    config = SPDConfig(hidden_size=32, depth=4, num_heads=4, vit_embed_dim=32,
                       vit_depth=1, vit_num_heads=4, vision_pool_num_heads=4)
    model = SPDPolicy(config).eval()
    dino = tmp_path / "dino.pt"
    torch.save(model.img_backbone.dinov3_model.state_dict(), dino)
    norm_stats = {
        "state": {"mean": np.linspace(-1, 1, 54).tolist(), "std": [0.5] * 54},
        "actions": {"mean": np.linspace(1, 2, 54).tolist(), "std": [0.3] * 54},
    }
    checkpoint = tmp_path / "last.pt"
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("img_backbone.")}
    torch.save({"policy": "spd", "architecture": SPD_ARCHITECTURE,
                "model_config": asdict(config), "model": state, "norm_stats": norm_stats,
                "dino_sha256": hashlib.sha256(dino.read_bytes()).hexdigest(),
                "ema": {"decay": 0.9, "model": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}}}, checkpoint)
    # run_metadata.json turns tuple camera keys into lists.
    config = SPDConfig(**json.loads(json.dumps(asdict(config))))
    engine = SPDInferencePolicy(checkpoint, SPDPolicyConfig(model=config, dino_checkpoint=str(dino), diffusion_steps=2), "cpu")
    obs = {"state": np.linspace(-0.3, 0.5, 54).astype(np.float32),
           "previous_actions": np.linspace(-0.4, 0.4, 54).astype(np.float32),
           "images": {k: np.full((3, 32, 48), 117, np.uint8) for k in ("top", "left_wrist")},
           "camera_validity": np.array([True, True, False])}
    return engine, obs


def test_inference_matches_normalized_model_and_reset_starts_new_episode(inference):
    engine, obs = inference
    noise = np.zeros((8, 54), dtype=np.float32)
    actual = engine.infer(obs, noise=noise)
    cache = engine.model.append_observation(
        None, torch.from_numpy(normalize(obs["state"], engine.norm_stats["state"]))[None],
        torch.from_numpy(normalize(obs["previous_actions"], engine.norm_stats["actions"]))[None],
        step=0, images={k: resize_pad_normalize_batch(torch.from_numpy(v)[None]) for k, v in obs["images"].items()},
        camera_validity=torch.from_numpy(obs["camera_validity"])[None],
    )
    normalized = engine.model.sample_actions_cached(cache, num_steps=2, noise=torch.from_numpy(noise)[None])
    expected = unnormalize(normalized[0].numpy(), engine.norm_stats["actions"])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    engine.observe({**obs, "state": obs["state"] + 0.2})
    engine.reset()
    with pytest.raises(ValueError, match="observe"):
        engine.infer(noise=noise)
    np.testing.assert_array_equal(engine.infer(obs, noise=noise), actual)


def test_invalid_camera_input_does_not_advance_cache(inference):
    engine, obs = inference
    engine.observe(obs)
    with pytest.raises(ValueError, match="missing valid camera"):
        engine.observe({**obs, "camera_validity": np.array([True, True, True])}, step=8)
    assert engine.cache.last_step == 0
    with pytest.raises(ValueError, match="prefix"):
        engine.infer(action_prefix=np.zeros((8, 54), np.float32))
    assert engine.cache.last_step == 0


def test_common_rollout_matches_per_tick_closed_loop_across_resets(inference, tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from abc_minimal.config import SimEvalConfig
    from abc_minimal.eval_policy import rollout_worlds
    from abc_sim.tianji_env import CAMERA_NAMES, TianjiTaskEnv
    from test_tianji_sim import _scene_files

    engine, _ = inference
    model_path, urdf_path = _scene_files(tmp_path)
    # Avoid command clipping hiding differences in the policy's feedback use.
    scene = ET.parse(model_path)
    for joint in scene.getroot().iter("joint"):
        joint.set("range", "-10 10")
    for actuator in scene.getroot().find("actuator"):
        actuator.set("ctrlrange", "-10 10")
    scene.write(model_path)
    urdf = ET.parse(urdf_path)
    for limit in urdf.getroot().iter("limit"):
        limit.set("lower", "-10")
        limit.set("upper", "10")
        limit.set("velocity", "600")
    urdf.write(urdf_path)
    # Only camera pixels are deterministic fixtures; both rollouts step real physics.
    pixels = {camera: np.full((3, 32, 48), 117, np.uint8) for camera in CAMERA_NAMES}
    monkeypatch.setattr(TianjiTaskEnv, "render_cameras", lambda self: pixels)
    kwargs = dict(model_path=model_path, urdf_path=urdf_path, height=32, width=48)
    reference_env = TianjiTaskEnv(**kwargs)
    actual_env = TianjiTaskEnv(**kwargs)
    config = SimEvalConfig(
        checkpoint="", policy="spd", embodiment="tianji_wuji2",
        num_worlds=2, num_chunks=2, execute_chunk_dim=8, prefix_length=0,
        rtc=False, fast_inference=False, diffusion_steps=2,
    )
    rng = np.random.default_rng(config.policy_seed)
    try:
        for world in range(config.num_worlds):
            observation = reference_env.reset(seed=config.seed + world)
            engine.reset()
            for chunk in range(config.num_chunks):
                noise = rng.standard_normal((8, 54), dtype=np.float32)
                actions = engine.infer(observation if chunk == 0 else None, noise=noise)
                for action in actions:
                    reference_env.step_one(action)
                    feedback = reference_env.obs()
                    engine.observe(feedback, step=feedback["step"])
        expected = reference_env.data.qpos.copy()
        worlds = rollout_worlds(config, engine, actual_env, 0, None, tmp_path, engine.model_config)
        assert [world["steps"] for world in worlds] == [16, 16]
        np.testing.assert_allclose(actual_env.data.qpos, expected, rtol=0, atol=1e-8)
    finally:
        reference_env.close()
        actual_env.close()
