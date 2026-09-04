import numpy as np
from types import SimpleNamespace

from spd_vr.config import TeleopConfig
from spd_vr.pico_hands import PicoHandFrame
from spd_vr.retarget_pair import WujiRetargetPair
from spd_vr.robot import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, RobotSpec
from spd_vr.teleop import PicoFrame, Side, TeleopMapper


class FakeArmIK:
    def __init__(self):
        self.calls = {Side.LEFT: 0, Side.RIGHT: 0}

    def solve(self, side, wrist_position, wrist_quaternion_xyzw, previous_qpos):
        self.calls[side] += 1
        return np.full(7, 0.1 * self.calls[side])


class FakeHands:
    def __init__(self):
        self.calls = {Side.LEFT: 0, Side.RIGHT: 0}

    def retarget(self, side, keypoints):
        self.calls[side] += 1
        return np.full(20, 0.1 * self.calls[side])

    def reset(self, side):
        self.calls[side] = 0


def _frame(sequence, valid=(True, True), epoch=1):
    quaternions = np.zeros((2, 4))
    quaternions[:, 3] = 1.0
    hands = np.zeros((2, 26, 7))
    hands[..., 6] = 1.0
    return PicoFrame(
        timestamp_ns=sequence * 10_000_000,
        sequence_id=sequence,
        tracking_epoch=epoch,
        wrist_position=np.zeros((2, 3)),
        wrist_quaternion_xyzw=quaternions,
        hands=hands,
        hand_scale=np.ones(2),
        valid=np.asarray(valid),
    )


def test_calibration_then_per_side_hold_preserves_last_safe_target():
    config = TeleopConfig(alignment_frames=2)
    robot = RobotSpec.from_urdf(config.urdf_path)
    mapper = TeleopMapper(robot, FakeArmIK(), FakeHands(), config)

    first, status = mapper.update(_frame(1))
    assert status.hold_reason == ("calibrating", "calibrating")
    assert not first.left_valid and not first.right_valid

    second, status = mapper.update(_frame(2))
    assert second.left_valid and second.right_valid
    left_safe = second.qpos[:27].copy()

    third, status = mapper.update(_frame(3, valid=(False, True)))
    assert not third.left_valid and third.right_valid
    assert status.hold_reason == ("tracking_invalid", "none")
    np.testing.assert_allclose(third.qpos[:27], left_safe)
    assert not np.array_equal(third.qpos[27:], second.qpos[27:])


def test_epoch_change_forces_recalibration():
    config = TeleopConfig(alignment_frames=2)
    mapper = TeleopMapper(
        RobotSpec.from_urdf(config.urdf_path), FakeArmIK(), FakeHands(), config
    )
    mapper.update(_frame(1))
    mapper.update(_frame(2))
    target, status = mapper.update(_frame(3, epoch=2))
    assert status.hold_reason == ("calibrating", "calibrating")
    assert not target.left_valid and not target.right_valid


def test_pico_frame_rejects_non_binary_validity_flags():
    with np.testing.assert_raises(ValueError):
        _frame(1, valid=(0.5, True))


def test_wuji_pair_holds_missing_or_malformed_hand_side_without_cross_contamination():
    class FakeRetargeter:
        def __init__(self, names):
            self.optimizer = SimpleNamespace(
                robot=SimpleNamespace(
                    dof_joint_names=names,
                    joint_limits=np.tile(np.asarray([[-1.0, 1.0]]), (20, 1)),
                )
            )

        def retarget(self, points):
            assert points.shape == (21, 3)
            return np.zeros(20)

    left = FakeRetargeter(list(LEFT_HAND_JOINTS))
    right = FakeRetargeter(list(RIGHT_HAND_JOINTS))
    pair = WujiRetargetPair(left, right)
    hand = np.zeros((26, 7), dtype=np.float64)
    hand[:, 6] = 1.0

    result = pair.retarget(
        {
            "left_hand": hand,
            "right_active": "not-a-boolean",
            "tracking_epoch": 1,
            "sequence_id": 1,
            "timestamp_ns": 1,
        }
    )
    assert result.left_valid is True
    assert result.right_valid is False
    assert result.right_hold_reason.value == "inactive"
    np.testing.assert_allclose(result.left_qpos, 0.0)


def test_wuji_pair_does_not_coerce_malformed_pico_frame_flag_to_active():
    class FakeRetargeter:
        def __init__(self, names):
            self.optimizer = SimpleNamespace(
                robot=SimpleNamespace(
                    dof_joint_names=names,
                    joint_limits=np.tile(np.asarray([[-1.0, 1.0]]), (20, 1)),
                )
            )

        def retarget(self, points):
            return np.zeros(20)

    hand = np.zeros((26, 7), dtype=np.float64)
    hand[:, 6] = 1.0
    pair = WujiRetargetPair(
        FakeRetargeter(list(LEFT_HAND_JOINTS)),
        FakeRetargeter(list(RIGHT_HAND_JOINTS)),
    )
    frame = PicoHandFrame(
        hand,
        hand,
        left_active="yes",
        right_active=True,
        tracking_epoch=1,
        sequence_id=1,
        timestamp_ns=1,
    )
    result = pair.retarget(frame)
    assert result.left_valid is False
    assert result.right_valid is True
