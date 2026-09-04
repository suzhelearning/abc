"""Canonical 54-DoF Tianji-Wuji2 joint contract.

The policy order is intentionally independent from the order in URDF or MJCF:
left arm, left hand, right arm, right hand.  All simulation access goes through
resolved qpos/dof/actuator addresses so a scene can safely add free joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import ROBOT_DOF


LEFT_ARM_JOINTS = tuple(f"Joint{i}_L" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"Joint{i}_R" for i in range(1, 8))


def _hand_joints(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_thumb_cmc_flex",
        f"{prefix}_thumb_cmc_abd",
        f"{prefix}_thumb_mcp",
        f"{prefix}_thumb_ip",
        f"{prefix}_index_finger_mcp_flex",
        f"{prefix}_index_finger_mcp_abd",
        f"{prefix}_index_finger_pip",
        f"{prefix}_index_finger_dip",
        f"{prefix}_middle_finger_mcp_flex",
        f"{prefix}_middle_finger_mcp_abd",
        f"{prefix}_middle_finger_pip",
        f"{prefix}_middle_finger_dip",
        f"{prefix}_ring_finger_mcp_flex",
        f"{prefix}_ring_finger_mcp_abd",
        f"{prefix}_ring_finger_pip",
        f"{prefix}_ring_finger_dip",
        f"{prefix}_pinky_mcp_flex",
        f"{prefix}_pinky_mcp_abd",
        f"{prefix}_pinky_pip",
        f"{prefix}_pinky_dip",
    )


LEFT_HAND_JOINTS = _hand_joints("l")
RIGHT_HAND_JOINTS = _hand_joints("r")
CANONICAL_JOINTS = (
    LEFT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
)


@dataclass(frozen=True, slots=True)
class RobotSpec:
    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        if self.joint_names != CANONICAL_JOINTS:
            raise ValueError("RobotSpec must use the canonical 54-DoF joint order")
        for name in ("lower", "upper", "velocity"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (ROBOT_DOF,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite [{ROBOT_DOF}] vector")
            object.__setattr__(self, name, value.copy())
        if np.any(self.lower >= self.upper):
            raise ValueError("joint limits must have lower < upper")
        if np.any(self.velocity <= 0):
            raise ValueError("velocity limits must be positive")

    @classmethod
    def from_urdf(cls, path: str | Path) -> "RobotSpec":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        joints = {node.get("name"): node for node in ET.parse(path).getroot().findall("joint")}
        missing = [name for name in CANONICAL_JOINTS if name not in joints]
        if missing:
            raise ValueError(f"URDF is missing canonical joints: {missing}")
        lower, upper, velocity = [], [], []
        for name in CANONICAL_JOINTS:
            joint = joints[name]
            if joint.get("type") != "revolute":
                raise ValueError(f"canonical joint {name} must be revolute")
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"canonical joint {name} has no limit")
            lower.append(float(limit.attrib["lower"]))
            upper.append(float(limit.attrib["upper"]))
            velocity.append(float(limit.attrib["velocity"]))
        return cls(CANONICAL_JOINTS, np.array(lower), np.array(upper), np.array(velocity))

    def clip(self, qpos: object) -> np.ndarray:
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (ROBOT_DOF,) or not np.all(np.isfinite(value)):
            raise ValueError(f"qpos must be a finite [{ROBOT_DOF}] vector")
        return np.clip(value, self.lower, self.upper)

    def resolve_mujoco(self, model: object) -> "MuJoCoAddresses":
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - dependency setup
            raise RuntimeError("MuJoCo is required to resolve robot addresses") from exc

        qpos, dof, actuator = [], [], []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing joint {name}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"MuJoCo joint {name} must be a hinge")
            qpos.append(int(model.jnt_qposadr[joint_id]))
            dof.append(int(model.jnt_dofadr[joint_id]))

            matches = [
                index
                for index in range(model.nu)
                if int(model.actuator_trnid[index, 0]) == joint_id
            ]
            if len(matches) != 1:
                raise ValueError(f"joint {name} must have exactly one actuator, got {matches}")
            actuator.append(matches[0])
        return MuJoCoAddresses(
            np.asarray(qpos, dtype=np.int64),
            np.asarray(dof, dtype=np.int64),
            np.asarray(actuator, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class MuJoCoAddresses:
    qpos: np.ndarray
    dof: np.ndarray
    actuator: np.ndarray

    def __post_init__(self) -> None:
        for name in ("qpos", "dof", "actuator"):
            value = np.asarray(getattr(self, name), dtype=np.int64)
            if value.shape != (ROBOT_DOF,) or len(np.unique(value)) != ROBOT_DOF:
                raise ValueError(f"{name} must contain {ROBOT_DOF} unique addresses")
            object.__setattr__(self, name, value.copy())

    def read_qpos(self, data: object) -> np.ndarray:
        return np.asarray(data.qpos, dtype=np.float64)[self.qpos].copy()

    def read_qvel(self, data: object) -> np.ndarray:
        return np.asarray(data.qvel, dtype=np.float64)[self.dof].copy()

    def write_control(self, data: object, target: object) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (ROBOT_DOF,) or not np.all(np.isfinite(value)):
            raise ValueError(f"target must be a finite [{ROBOT_DOF}] vector")
        data.ctrl[self.actuator] = value


__all__ = [
    "CANONICAL_JOINTS",
    "LEFT_ARM_JOINTS",
    "LEFT_HAND_JOINTS",
    "RIGHT_ARM_JOINTS",
    "RIGHT_HAND_JOINTS",
    "MuJoCoAddresses",
    "RobotSpec",
]
