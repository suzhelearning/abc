"""MuJoCo damped-least-squares IK for the two Tianji wrists."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .robot import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, RobotSpec
from .teleop import Side
from .qp_arm import ArmQPSolver


class MuJoCoArmIK:
    """Resolve each seven-joint arm independently in the 14-DoF projection."""

    def __init__(
        self,
        model_path: str | Path,
        robot: RobotSpec,
        *,
        iterations: int = 20,
        damping: float = 1e-3,
        step_size: float = 0.7,
        use_qp: bool = True,
        control_hz: int = 200,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - dependency setup
            raise RuntimeError("MuJoCo is required for arm IK") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.robot = robot
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.use_qp = bool(use_qp)
        self.control_hz = int(control_hz)
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._joints: dict[Side, np.ndarray] = {}
        self._dofs: dict[Side, np.ndarray] = {}
        self._sites: dict[Side, int] = {}
        self._limits: dict[Side, tuple[np.ndarray, np.ndarray]] = {}
        for side, names, limits in (
            (Side.LEFT, LEFT_ARM_JOINTS, (robot.lower[:7], robot.upper[:7])),
            (Side.RIGHT, RIGHT_ARM_JOINTS, (robot.lower[27:34], robot.upper[27:34])),
        ):
            joint_ids = np.asarray(
                [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            )
            if np.any(joint_ids < 0):
                raise ValueError(f"arm IK model is missing {side.value} joints")
            self._joints[side] = np.asarray(self.model.jnt_qposadr[joint_ids], dtype=np.int64)
            self._dofs[side] = np.asarray(self.model.jnt_dofadr[joint_ids], dtype=np.int64)
            self._sites[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{'l' if side is Side.LEFT else 'r'}_wrist_target"
            )
            if self._sites[side] < 0:
                raise ValueError(f"arm IK model has no wrist site for {side.value}")
            self._limits[side] = limits
        mujoco.mj_forward(self.model, self.data)
        self._neutral_position = {
            side: self.data.site_xpos[site].copy() for side, site in self._sites.items()
        }
        self._neutral_rotation = {
            side: self.data.site_xmat[site].reshape(3, 3).copy()
            for side, site in self._sites.items()
        }
        self._qp: dict[Side, ArmQPSolver] = {}
        if self.use_qp:
            for side, names, limits in (
                (Side.LEFT, LEFT_ARM_JOINTS, (robot.lower[:7], robot.upper[:7])),
                (Side.RIGHT, RIGHT_ARM_JOINTS, (robot.lower[27:34], robot.upper[27:34])),
            ):
                joint_ids = tuple(
                    int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
                    for name in names
                )
                if any(index < 0 for index in joint_ids):
                    raise ValueError(f"arm IK model is missing {side.value} joints")
                arm_limits = np.column_stack(limits)
                self._qp[side] = ArmQPSolver(
                    self.model,
                    self.data,
                    side=side.value,
                    site_name=f"{'l' if side is Side.LEFT else 'r'}_wrist_target",
                    joint_ids=joint_ids,
                    position_limits=arm_limits,
                    velocity_limits=robot.velocity[:7] if side is Side.LEFT else robot.velocity[27:34],
                )

    @staticmethod
    def _xyzw_matrix(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = quaternion
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        skew = 0.5 * (target @ current.T - current @ target.T)
        return np.asarray((skew[2, 1], skew[0, 2], skew[1, 0]))

    def solve(
        self,
        side: Side,
        wrist_position: np.ndarray,
        wrist_quaternion_xyzw: np.ndarray,
        previous_qpos: np.ndarray,
    ) -> np.ndarray:
        wrist_position = np.asarray(wrist_position, dtype=np.float64)
        wrist_quaternion_xyzw = np.asarray(wrist_quaternion_xyzw, dtype=np.float64)
        previous_qpos = np.asarray(previous_qpos, dtype=np.float64)
        if wrist_position.shape != (3,) or wrist_quaternion_xyzw.shape != (4,) or previous_qpos.shape != (7,):
            raise ValueError("arm IK inputs must be position[3], quaternion[4], and qpos[7]")
        if not np.all(np.isfinite(wrist_position)) or not np.all(np.isfinite(wrist_quaternion_xyzw)) or not np.all(np.isfinite(previous_qpos)):
            raise ValueError("arm IK inputs must be finite")
        norm = float(np.linalg.norm(wrist_quaternion_xyzw))
        if norm <= 1.0e-12:
            raise ValueError("wrist quaternion must be non-zero")
        wrist_quaternion_xyzw = wrist_quaternion_xyzw / norm
        if self.use_qp:
            target_pose = np.eye(4, dtype=np.float64)
            target_pose[:3, 3] = self._neutral_position[side] + wrist_position
            target_pose[:3, :3] = self._neutral_rotation[side] @ self._xyzw_matrix(
                wrist_quaternion_xyzw
            )
            result = self._qp[side].solve(
                previous_qpos,
                target_pose,
                1.0 / float(self.control_hz),
            )
            if not result.success:
                raise ValueError(f"arm QP failed: {result.status}")
            arm = previous_qpos + np.asarray(result.dq, dtype=np.float64) / float(self.control_hz)
            lower, upper = self._limits[side]
            arm = np.clip(arm, lower, upper)
            if not np.all(np.isfinite(arm)):
                raise ValueError("arm QP produced a non-finite target")
            return arm
        qpos_addresses = self._joints[side]
        dof_addresses = self._dofs[side]
        lower, upper = self._limits[side]
        self.data.qpos[qpos_addresses] = np.clip(previous_qpos, lower, upper)
        target_position = self._neutral_position[side] + wrist_position
        target_rotation = self._neutral_rotation[side] @ self._xyzw_matrix(
            wrist_quaternion_xyzw
        )
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        site = self._sites[side]
        for _ in range(self.iterations):
            self.mujoco.mj_forward(self.model, self.data)
            position_error = target_position - self.data.site_xpos[site]
            rotation_error = self._rotation_error(
                self.data.site_xmat[site].reshape(3, 3), target_rotation
            )
            error = np.concatenate((position_error, rotation_error))
            if np.linalg.norm(error) < 1e-4:
                break
            self.mujoco.mj_jacSite(
                self.model, self.data, jacobian_position, jacobian_rotation, site
            )
            jacobian = np.vstack((jacobian_position[:, dof_addresses], jacobian_rotation[:, dof_addresses]))
            system = jacobian @ jacobian.T + self.damping * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)
            self.data.qpos[qpos_addresses] = np.clip(
                self.data.qpos[qpos_addresses] + self.step_size * delta, lower, upper
            )
        result = np.asarray(self.data.qpos[qpos_addresses], dtype=np.float64).copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("arm IK produced a non-finite target")
        return result


__all__ = ["MuJoCoArmIK"]
