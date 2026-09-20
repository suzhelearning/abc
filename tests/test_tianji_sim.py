"""Real CPU MuJoCo contracts; camera rendering is deliberately not exercised."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from abc_sim.tianji_env import (
    CAMERA_NAMES,
    CANONICAL_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    TianjiTaskEnv,
)


def _scene_files(tmp_path):
    scene = ET.Element("mujoco")
    ET.SubElement(scene, "compiler", angle="radian")
    ET.SubElement(scene, "option", timestep=str(1 / 480), integrator="implicitfast")
    world = ET.SubElement(scene, "worldbody")
    # The free joint deliberately occupies qpos[0:7] and qvel[0:6].
    hammer = ET.SubElement(world, "body", name="hammer", pos="0 0 0.3")
    ET.SubElement(hammer, "freejoint", name="hammer_free")
    ET.SubElement(hammer, "geom", type="sphere", size="0.02", mass="0.05")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1")
    ET.SubElement(world, "geom", name="raised_floor", type="box", pos="2 0 0.8", size="0.1 0.1 0.1")
    for camera in CAMERA_NAMES:
        ET.SubElement(world, "camera", name=camera, pos="0 -3 2")
    urdf = ET.Element("robot", name="synthetic_tianji")
    ET.SubElement(urdf, "link", name="world")

    def add_joint(parent, name, pos, *, arm_plate=False):
        body_name = f"body_{name}"
        body = ET.SubElement(parent, "body", name=body_name, pos=pos)
        ET.SubElement(body, "joint", name=name, type="hinge", axis="0 0 1", range="-1 1", limited="true", armature="0.1")
        if arm_plate:
            ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1", mass="0.1")
        else:
            ET.SubElement(body, "geom", type="sphere", size="0.015", mass="0.05")
        ET.SubElement(urdf, "link", name=body_name)
        joint = ET.SubElement(urdf, "joint", name=name, type="revolute")
        ET.SubElement(joint, "parent", link=parent.get("name", "world"))
        ET.SubElement(joint, "child", link=body_name)
        ET.SubElement(joint, "limit", lower="-1", upper="1", velocity="0.6", effort="10")

    for index, name in enumerate(reversed(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)):
        add_joint(world, name, "3 0 0.8" if name == "Joint1_L" else f"{10 + index * 0.2} 0 1", arm_plate=name == "Joint1_L")
    for side, names, x in (("l", LEFT_HAND_JOINTS, 0), ("r", RIGHT_HAND_JOINTS, 5)):
        palm_name = f"{side}_wrist"
        palm = ET.SubElement(world, "body", name=palm_name, pos=f"{x} 0 0.8")
        ET.SubElement(palm, "geom", type="box", size="0.1 0.1 0.1")
        ET.SubElement(urdf, "link", name=palm_name)
        for index, name in enumerate(reversed(names)):
            add_joint(palm, name, f"{0.5 + index * 0.12} 1 0")
    actuator = ET.SubElement(scene, "actuator")
    for name in reversed(CANONICAL_JOINTS):
        ET.SubElement(actuator, "position", name=f"servo_{name}", joint=name, kp="1", kv="0.1", ctrlrange="-1 1", ctrllimited="true")
    model_path = tmp_path / "scene.xml"
    urdf_path = tmp_path / "robot.urdf"
    ET.ElementTree(scene).write(model_path)
    ET.ElementTree(urdf).write(urdf_path)
    return model_path, urdf_path


@pytest.fixture
def make_env(tmp_path, monkeypatch):
    model_path, urdf_path = _scene_files(tmp_path)
    # Physics tests do not create GL contexts or replace any physics engine.
    monkeypatch.setattr(TianjiTaskEnv, "render_cameras", lambda self: {})
    environments = []

    def make(**kwargs):
        env = TianjiTaskEnv(model_path=model_path, urdf_path=urdf_path, height=16, width=16, **kwargs)
        environments.append(env)
        return env

    yield make
    for env in environments:
        env.close()


def test_named_mapping_previous_measured_feedback_and_clock(make_env):
    initial = np.linspace(-0.2, 0.2, 54)
    env = make_env(initial_qpos=initial)
    before = env.reset(seed=4)
    np.testing.assert_allclose(before["state"], initial, atol=1e-8)
    np.testing.assert_array_equal(before["previous_actions"], before["state"])
    np.testing.assert_array_equal(before["camera_validity"], [True, True, False])
    np.testing.assert_allclose(env.data.qpos[:3], [0, 0, 0.3])
    for name, position in zip(CANONICAL_JOINTS, initial, strict=True):
        assert env.data.ctrl[env.model.actuator(f"servo_{name}").id] == position
    env.step_one(initial + 0.01)
    first = env.obs()
    np.testing.assert_array_equal(first["previous_actions"], before["state"])
    assert np.any(np.abs(first["state"] - before["state"]) > 1e-7)
    assert not np.allclose(first["state"], initial + 0.01, atol=1e-4)
    assert first["step"] == 1
    assert first["images"] == {}
    assert env.data.time == pytest.approx(1 / 30, abs=1e-12)
    assert env.evaluate()["tracking_rmse"] > 0
    env.step_one(initial + 0.02)
    np.testing.assert_array_equal(env.obs()["previous_actions"], first["state"])
    assert env.data.time == pytest.approx(2 / 30, abs=1e-12)
    reset = env.reset(seed=7)
    np.testing.assert_allclose(reset["state"], initial, atol=1e-8)
    np.testing.assert_array_equal(reset["previous_actions"], reset["state"])
    assert env.data.time == 0
    assert env.evaluate()["tracking_rmse_episode"] == 0
    assert env.randomization == {"mode": "fixed_scene", "seed": 7, "randomized": False}


def test_model_default_is_reset_feedback_not_old_command_history(make_env):
    env = make_env()
    obs = env.reset(seed=0)
    expected = [env.model.qpos0[env.model.joint(name).qposadr[0]] for name in CANONICAL_JOINTS]
    np.testing.assert_array_equal(obs["state"], expected)
    np.testing.assert_array_equal(obs["previous_actions"], expected)


@pytest.mark.parametrize("bad", [np.full(54, np.nan), np.full(54, np.inf), np.zeros(53), np.zeros((1, 54))])
def test_rejects_nonfinite_or_wrong_shape_without_advancing(make_env, bad):
    env = make_env()
    env.reset(seed=0)
    qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
    with pytest.raises(ValueError, match="finite \\[54\\]"):
        env.step_one(bad)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.ctrl, ctrl)
    assert env.data.time == 0


def test_limits_are_around_previous_target_not_lagging_measurement(make_env):
    env = make_env(initial_qpos=np.full(54, 0.98))
    env.reset(seed=0)
    env.step_one(np.full(54, 100.0))
    np.testing.assert_allclose(env.data.ctrl, 1.0, atol=1e-12)
    assert env.evaluate()["joint_bound_clips"] == 54
    env.step_one(np.full(54, -100.0))
    np.testing.assert_allclose(env.data.ctrl, 0.98, atol=1e-12)
    assert env.evaluate()["velocity_limit_clips"] == 54
    env.step_one(np.full(54, -100.0))
    np.testing.assert_allclose(env.data.ctrl, 0.96, atol=1e-12)
    assert env.evaluate()["limit_clips_total"] == 162
    assert env.evaluate()["limit_clip_fraction"] == 1
    assert not np.allclose(env.obs()["state"], 0.96, atol=1e-3)
    env.reset(seed=0)
    assert env.evaluate()["limit_clips_total"] == 0


def _place_object(env, *, x, z):
    # Explicit test-state setup, followed by real contact solving and integration.
    joint = env.model.joint("hammer_free")
    qpos = int(joint.qposadr[0])
    dof = int(joint.dofadr[0])
    env.data.qpos[qpos:qpos + 3] = [x, 0, z]
    env.data.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(env.model, env.data)


def test_lift_contact_hold_is_tick_based_and_ever_success_latches(make_env):
    env = make_env(hold_steps=3)
    env.reset(seed=0)
    assert not env.evaluate()["success"]
    assert env.evaluate()["reward"] == 0
    _place_object(env, x=0, z=0.9199)
    for tick in range(1, 4):
        env.step_one(np.zeros(54))
        result = env.evaluate()
        assert result["hand_object_contact"]
        assert result["object_height_gain"] >= env.lift_height
        assert result["hold_count"] == tick
        assert result["success"] == (tick == 3)
        assert result["reward"] == 1
        assert env.evaluate() == result
    _place_object(env, x=1, z=0.92)
    env.step_one(np.zeros(54))
    result = env.evaluate()
    assert not result["hand_object_contact"]
    assert not result["success"]
    assert result["ever_success"]
    assert result["hold_count"] == 0
    env.reset(seed=0)
    assert not env.evaluate()["ever_success"]
    assert env.evaluate()["hold_count"] == 0


@pytest.mark.parametrize("support_x", [2, 3])
def test_floor_and_arm_contact_do_not_count_as_grasp(make_env, support_x):
    env = make_env(hold_steps=1)
    env.reset(seed=0)
    _place_object(env, x=support_x, z=0.9199)
    env.step_one(np.zeros(54))
    assert env.data.ncon > 0
    result = env.evaluate()
    assert result["object_height_gain"] >= env.lift_height
    assert not result["hand_object_contact"]
    assert not result["success"]


def test_rejects_bad_initialization_and_unimplemented_reset_options(make_env):
    with pytest.raises(ValueError, match="initial_qpos"):
        make_env(initial_qpos=np.full(54, 1.001))
    with pytest.raises(ValueError, match="initial_qpos"):
        make_env(initial_qpos=np.full(54, np.nan))
    env = make_env()
    with pytest.raises(ValueError, match="prepared scene is fixed"):
        env.reset(seed=0, options={"randomize": True})


@pytest.mark.parametrize("fault", ["time", "qpos", "qvel", "qacc", "BADQPOS", "BADQVEL", "BADQACC", "CONTACTFULL"])
def test_physics_faults_raise_and_require_reset(make_env, fault):
    env = make_env()
    env.reset(seed=0)
    if fault == "time":
        env.data.time = 0.1
    elif fault.startswith("BAD") or fault == "CONTACTFULL":
        env.data.warning[getattr(mujoco.mjtWarning, f"mjWARN_{fault}")].number += 1
    else:
        getattr(env.data, fault)[0] = np.nan
    with pytest.raises(RuntimeError, match="MuJoCo"):
        env.step_one(np.zeros(54))
    with pytest.raises(RuntimeError, match="reset required"):
        env.obs()
    env.reset(seed=0)
    env.step_one(np.zeros(54))
    assert env.data.time == pytest.approx(1 / 30)


@pytest.mark.parametrize("mutation", ["timestep", "gear", "range", "motor", "duplicate", "fixed_object"])
def test_rejects_incompatible_scene_contract(tmp_path, mutation):
    model_path, urdf_path = _scene_files(tmp_path)
    tree = ET.parse(model_path)
    root = tree.getroot()
    actuator = root.find("actuator")[0]
    if mutation == "timestep":
        root.find("option").set("timestep", "0.002")
    elif mutation == "gear":
        actuator.set("gear", "2")
    elif mutation == "range":
        actuator.set("ctrlrange", "-0.5 0.5")
    elif mutation == "motor":
        actuator.tag = "motor"
        del actuator.attrib["kp"]
        del actuator.attrib["kv"]
    elif mutation == "duplicate":
        ET.SubElement(root.find("actuator"), "position", joint=actuator.attrib["joint"], kp="1", ctrlrange="-1 1", ctrllimited="true")
    else:
        hammer = root.find("worldbody/body")
        hammer.remove(hammer.find("freejoint"))
    tree.write(model_path)
    with pytest.raises(ValueError):
        TianjiTaskEnv(model_path=model_path, urdf_path=urdf_path, height=16, width=16)
