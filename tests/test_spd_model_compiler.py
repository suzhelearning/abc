import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from spd_vr.config import TeleopConfig
from spd_vr.data import EpisodeWriter, validate_episode
from spd_vr.ik import MuJoCoArmIK
from spd_vr.model_compiler.artifacts import (
    ArtifactError,
    compile_models,
    verify_artifacts,
    verify_contact_qualified,
)
import pytest
from spd_vr.robot import RobotSpec
from spd_vr.simulation import SPDVRSim
from spd_vr.teleop import Side
from spd_vr.scenes.model_scene import write_scene_model
from spd_vr.scenes.registry import get_task


def test_raw_collision_smoke_build_has_54d_and_three_policy_cameras(tmp_path):
    urdf = Path(TeleopConfig().urdf_path)
    result = compile_models(
        urdf, tmp_path / "generated", tmp_path / "collision_cache", raw_collisions=True
    )
    verified = verify_artifacts(result.path, urdf)
    with pytest.raises(ArtifactError, match="raw collision"):
        verify_contact_qualified(result.collision_manifest, urdf_path=urdf)
    robot = RobotSpec.from_urdf(urdf)
    with SPDVRSim(verified.full_model, robot) as simulation:
        assert simulation.model.nq == 54
        assert simulation.model.nu == 54
        assert simulation.model.ncam == 3
        frames = simulation.render_cameras()
        assert frames["top"].rgb.shape == (168, 224, 3)
        assert frames["left_wrist"].segmentation.shape == (168, 224, 2)
        benchmark = simulation.benchmark(1.0 / 60.0)
        assert benchmark["control_ticks"] == 1
        assert benchmark["physics_steps"] == 8
        assert benchmark["realtime_budget_ms"] == pytest.approx(1000.0 / 60.0)
        episode = tmp_path / "simulated.h5"
        hands = np.zeros((2, 26, 7), dtype=np.float32)
        hands[..., 6] = 1.0
        with EpisodeWriter(episode, {"source": "raw_mujoco_smoke"}) as writer:
            for index in range(3):
                simulation.step()
                writer.append(
                    simulation.record_frame(
                        1_000_000_000 + index + 1,
                        hands,
                        2_000_000_000 + index,
                        3_000_000_000 + index,
                        index + 1,
                        1,
                        np.ones(2, dtype=np.float32),
                        np.ones(2, dtype=np.bool_),
                    )
                )
        assert validate_episode(episode, verify_checksums=True)["raw_frames"] == 3
    ik = MuJoCoArmIK(verified.arm_model, robot, iterations=1)
    result = ik.solve(Side.LEFT, [0, 0, 0], [0, 0, 0, 1], [0] * 7)
    assert result.shape == (7,)

    scene_result = get_task("cups/pyramid").reset(3)
    scene_model = write_scene_model(
        verified.full_model, scene_result, tmp_path / "scene.xml"
    )
    with SPDVRSim(scene_model, robot) as scene_sim:
        scene_sim.reset_scene(scene_result)
        assert scene_sim.model.nq > 54 and scene_sim.model.nu == 54
        assert len(scene_sim.object_state()) == len(scene_result.objects)
