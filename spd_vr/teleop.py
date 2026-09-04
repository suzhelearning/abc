"""Simulation-only PICO-to-Tianji-Wuji2 command mapping.

This module deliberately exposes no real-robot transport.  A caller supplies
one arm IK solver and one hand retargeter per side; failures only hold the
affected side at its last safe target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Protocol

import numpy as np

from .config import TeleopConfig
from .contracts import ARM_DOF_PER_SIDE, HAND_DOF_PER_SIDE, JointTarget
from .robot import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS, RobotSpec


PICO_JOINT_COUNT = 26
PICO_TO_MEDIAPIPE = np.asarray(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    dtype=np.int64,
)


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class ArmIK(Protocol):
    def solve(
        self,
        side: Side,
        wrist_position: np.ndarray,
        wrist_quaternion_xyzw: np.ndarray,
        previous_qpos: np.ndarray,
    ) -> np.ndarray: ...


class HandRetargeter(Protocol):
    def retarget(self, side: Side, keypoints: np.ndarray) -> np.ndarray: ...

    def reset(self, side: Side) -> None: ...


@dataclass(frozen=True, slots=True)
class PicoFrame:
    timestamp_ns: int
    sequence_id: int
    tracking_epoch: int
    wrist_position: np.ndarray
    wrist_quaternion_xyzw: np.ndarray
    hands: np.ndarray
    hand_scale: np.ndarray
    valid: np.ndarray
    source_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        shapes = {
            "wrist_position": (2, 3),
            "wrist_quaternion_xyzw": (2, 4),
            "hands": (2, PICO_JOINT_COUNT, 7),
            "hand_scale": (2,),
            "valid": (2,),
        }
        for name, shape in shapes.items():
            if name == "valid":
                raw = np.asarray(getattr(self, name))
                if raw.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
                if raw.dtype.kind == "b":
                    value = raw.astype(np.bool_, copy=False)
                elif raw.dtype.kind in {"i", "u"} and np.all(np.isin(raw, (0, 1))):
                    value = raw.astype(np.bool_, copy=False)
                else:
                    raise ValueError("valid must contain only boolean or 0/1 integer flags")
            else:
                value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if name != "valid" and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, value.copy())
        source_timestamp_ns = (
            self.timestamp_ns
            if self.source_timestamp_ns is None
            else self.source_timestamp_ns
        )
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, Integral)
            or not 0 <= int(self.timestamp_ns) <= 0x7FFFFFFFFFFFFFFF
            or isinstance(source_timestamp_ns, bool)
            or not isinstance(source_timestamp_ns, Integral)
            or not 0 <= int(source_timestamp_ns) <= 0x7FFFFFFFFFFFFFFF
            or isinstance(self.sequence_id, bool)
            or not isinstance(self.sequence_id, Integral)
            or not 0 <= int(self.sequence_id) <= 0xFFFFFFFFFFFFFFFF
            or isinstance(self.tracking_epoch, bool)
            or not isinstance(self.tracking_epoch, Integral)
            or not 0 <= int(self.tracking_epoch) <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError("PICO counters and timestamps must be non-negative")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "source_timestamp_ns", int(source_timestamp_ns))
        object.__setattr__(self, "sequence_id", int(self.sequence_id))
        object.__setattr__(self, "tracking_epoch", int(self.tracking_epoch))
        norms = np.linalg.norm(self.wrist_quaternion_xyzw, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-2):
            raise ValueError("wrist quaternions must be normalized")
        if np.any(self.hand_scale <= 0):
            raise ValueError("hand_scale must be positive")


@dataclass(frozen=True, slots=True)
class TeleopStatus:
    calibrated: tuple[bool, bool]
    hold_reason: tuple[str, str]


class StablePoseCalibration:
    """Collect a bounded neutral pose before a side may be controlled."""

    def __init__(self, config: TeleopConfig) -> None:
        self.config = config
        self.positions: deque[np.ndarray] = deque(maxlen=config.alignment_frames)
        self.quaternions: deque[np.ndarray] = deque(maxlen=config.alignment_frames)
        self.position: np.ndarray | None = None
        self.quaternion: np.ndarray | None = None

    def reset(self) -> None:
        self.positions.clear()
        self.quaternions.clear()
        self.position = None
        self.quaternion = None

    def add(self, position: np.ndarray, quaternion: np.ndarray) -> bool:
        if self.position is not None:
            return True
        self.positions.append(position.copy())
        q = quaternion.copy()
        if self.quaternions and np.dot(q, self.quaternions[0]) < 0:
            q = -q
        self.quaternions.append(q)
        if len(self.positions) < self.config.alignment_frames:
            return False
        positions = np.stack(self.positions)
        quaternions = np.stack(self.quaternions)
        q_mean = quaternions.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean)
        stable_position = float(np.max(np.std(positions, axis=0))) <= self.config.alignment_position_std_m
        stable_rotation = bool(
            np.all(np.abs(quaternions @ q_mean) >= self.config.alignment_quaternion_dot_min)
        )
        if not (stable_position and stable_rotation):
            self.positions.popleft()
            self.quaternions.popleft()
            return False
        self.position = positions.mean(axis=0)
        self.quaternion = q_mean
        return True


# Keep the private name for downstream code that imported the early prototype.
_Calibration = StablePoseCalibration


class WujiRetargetAdapter:
    """Load the vendored Wuji retargeters and convert PICO 26→21 positions."""

    def __init__(self, left_config: str, right_config: str) -> None:
        from wuji_retargeting import Retargeter

        self._retargeters = {
            Side.LEFT: Retargeter.from_yaml(left_config, hand_side="left"),
            Side.RIGHT: Retargeter.from_yaml(right_config, hand_side="right"),
        }
        self._permutations = {}
        for side, canonical in (
            (Side.LEFT, LEFT_HAND_JOINTS),
            (Side.RIGHT, RIGHT_HAND_JOINTS),
        ):
            source = list(self._retargeters[side].optimizer.robot.dof_joint_names)
            if len(source) != HAND_DOF_PER_SIDE or set(source) != set(canonical):
                raise ValueError(f"Wuji {side.value} joint names do not match the robot contract")
            self._permutations[side] = np.asarray(
                [source.index(name) for name in canonical], dtype=np.int64
            )

    def retarget(self, side: Side, keypoints: np.ndarray) -> np.ndarray:
        positions = np.asarray(keypoints, dtype=np.float64)
        if positions.shape != (PICO_JOINT_COUNT, 7):
            raise ValueError("PICO hand must have shape [26,7]")
        # Retargeter performs the wrist-frame transform exactly once.
        value = self._retargeters[side].retarget(positions[PICO_TO_MEDIAPIPE, :3])
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (HAND_DOF_PER_SIDE,) or not np.all(np.isfinite(value)):
            raise ValueError("Wuji retargeter returned an invalid 20-DoF vector")
        return value[self._permutations[side]]

    def reset(self, side: Side) -> None:
        self._retargeters[side].reset_filter()


class TeleopMapper:
    """Build canonical 54-D commands with per-side calibration and HOLD."""

    def __init__(
        self,
        robot: RobotSpec,
        arm_ik: ArmIK,
        hands: HandRetargeter,
        config: TeleopConfig | None = None,
        initial_qpos: np.ndarray | None = None,
    ) -> None:
        self.robot = robot
        self.arm_ik = arm_ik
        self.hands = hands
        self.config = config or TeleopConfig()
        initial = np.zeros(54) if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float64)
        self._target = self.robot.clip(initial)
        self._calibration = [StablePoseCalibration(self.config), StablePoseCalibration(self.config)]
        self._epoch: int | None = None
        self._sequence: int | None = None

    def reset(self) -> None:
        for index, side in enumerate((Side.LEFT, Side.RIGHT)):
            self._calibration[index].reset()
            self.hands.reset(side)
        self._sequence = None

    @staticmethod
    def _slices(index: int) -> tuple[slice, slice]:
        if index == 0:
            return slice(0, 7), slice(7, 27)
        return slice(27, 34), slice(34, 54)

    @staticmethod
    def _relative_quaternion(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return current * inverse(reference), using PICO's xyzw order."""
        x1, y1, z1, w1 = current
        x2, y2, z2, w2 = (-reference[0], -reference[1], -reference[2], reference[3])
        result = np.asarray(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dtype=np.float64,
        )
        return result / np.linalg.norm(result)

    def update(self, frame: PicoFrame, *, now_ns: int | None = None) -> tuple[JointTarget, TeleopStatus]:
        if not isinstance(frame, PicoFrame):
            raise TypeError("frame must be a PicoFrame")
        if self._epoch != frame.tracking_epoch:
            self._epoch = frame.tracking_epoch
            self.reset()
        stale_sequence = self._sequence is not None and frame.sequence_id <= self._sequence
        self._sequence = max(frame.sequence_id, self._sequence or 0)
        now_ns = frame.timestamp_ns if now_ns is None else int(now_ns)
        age_ns = now_ns - frame.timestamp_ns
        stale_time = not 0 <= age_ns <= int(self.config.stale_after_ms * 1e6)

        valid_out = [False, False]
        reasons = ["hold", "hold"]
        for index, side in enumerate((Side.LEFT, Side.RIGHT)):
            arm_slice, hand_slice = self._slices(index)
            if stale_sequence or stale_time:
                reasons[index] = "stale"
                continue
            if not frame.valid[index]:
                reasons[index] = "tracking_invalid"
                continue
            calibration = self._calibration[index]
            if not calibration.add(
                frame.wrist_position[index], frame.wrist_quaternion_xyzw[index]
            ):
                reasons[index] = "calibrating"
                continue
            try:
                arm = np.asarray(
                    self.arm_ik.solve(
                        side,
                        frame.wrist_position[index] - calibration.position,
                        self._relative_quaternion(
                            frame.wrist_quaternion_xyzw[index], calibration.quaternion
                        ),
                        self._target[arm_slice],
                    ),
                    dtype=np.float64,
                )
                hand_points = frame.hands[index].copy()
                hand_points[:, :3] *= frame.hand_scale[index]
                hand = np.asarray(self.hands.retarget(side, hand_points), dtype=np.float64)
                if arm.shape != (ARM_DOF_PER_SIDE,) or hand.shape != (HAND_DOF_PER_SIDE,):
                    raise ValueError("solver output shape mismatch")
                if not np.all(np.isfinite(arm)) or not np.all(np.isfinite(hand)):
                    raise ValueError("solver output is non-finite")
                candidate = self._target.copy()
                desired = np.concatenate((arm, hand))
                side_slice = slice(arm_slice.start, hand_slice.stop)
                previous = self._target[side_slice]
                max_delta = self.robot.velocity[side_slice] / self.config.control_hz
                candidate[side_slice] = previous + np.clip(
                    desired - previous, -max_delta, max_delta
                )
                self._target = self.robot.clip(candidate)
                valid_out[index] = True
                reasons[index] = "none"
            except Exception:
                reasons[index] = "solver_failure"

        return (
            JointTarget(frame.timestamp_ns, self._target, valid_out[0], valid_out[1]),
            TeleopStatus(
                (self._calibration[0].position is not None, self._calibration[1].position is not None),
                (reasons[0], reasons[1]),
            ),
        )


__all__ = [
    "ArmIK",
    "HandRetargeter",
    "PICO_JOINT_COUNT",
    "PICO_TO_MEDIAPIPE",
    "PicoFrame",
    "Side",
    "StablePoseCalibration",
    "TeleopMapper",
    "TeleopStatus",
    "WujiRetargetAdapter",
]
