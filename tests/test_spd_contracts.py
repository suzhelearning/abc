import numpy as np
import pytest

from spd_vr.config import SPDModelConfig, validate_spd_model_config
from spd_vr.contracts import ROBOT_DOF
from spd_vr.robot import (
    CANONICAL_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
    RobotSpec,
)


def test_authoritative_urdf_has_exact_canonical_contract(vendor_urdf):
    path = vendor_urdf
    robot = RobotSpec.from_urdf(path)
    assert path.is_file()
    assert len(robot.joint_names) == ROBOT_DOF
    assert robot.joint_names == (
        LEFT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
    )
    assert robot.joint_names == CANONICAL_JOINTS
    assert np.all(robot.lower < robot.upper)


def test_robot_clip_rejects_bad_shape_and_clips_limits(vendor_urdf):
    robot = RobotSpec.from_urdf(vendor_urdf)
    clipped = robot.clip(np.full(ROBOT_DOF, 100.0))
    np.testing.assert_allclose(clipped, robot.upper)
    try:
        robot.clip(np.zeros(53))
    except ValueError as exc:
        assert "54" in str(exc)
    else:
        raise AssertionError("bad joint vector was accepted")


def test_spd_image_stride_is_fixed_by_contract():
    with pytest.raises(ValueError, match="image_stride"):
        validate_spd_model_config(SPDModelConfig(image_stride=4))
