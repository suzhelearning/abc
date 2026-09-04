"""Single-arm MuJoCo Jacobian velocity QP.

The workspace is created once and updated in place at 200 Hz.  This keeps
solver allocation out of the callback/tick path while enforcing both URDF
position limits and velocity limits.  It only returns simulation joint
velocity; no physical robot transport is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import osqp
from scipy import sparse

try:
    import mujoco
except ImportError:  # pragma: no cover - dependency setup
    mujoco = None  # type: ignore[assignment]

from .alignment import _pose_matrix


@dataclass(frozen=True, slots=True)
class ArmSolveResult:
    dq: np.ndarray
    success: bool
    status: str
    position_error_m: float = math.inf
    orientation_error_rad: float = math.inf

    def __post_init__(self) -> None:
        value = np.array(self.dq, dtype=np.float64, copy=True).reshape(-1)
        value.setflags(write=False)
        object.__setattr__(self, "dq", value)

    @property
    def solved(self) -> bool:
        return self.success

    @property
    def failure(self) -> bool:
        return not self.success


@dataclass(frozen=True, slots=True)
class QPConfig:
    position_weight: float = 1.0
    orientation_weight: float = 0.5
    lambda_damp: float = 1.0e-4
    lambda_home: float = 1.0e-3
    home_gain: float = 1.0
    max_linear_speed: float = 2.0
    max_angular_speed: float = 4.0
    max_position_error_m: float = 2.0
    max_orientation_error_rad: float = math.pi


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-8:
        return 0.5 * np.asarray(
            [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
            dtype=np.float64,
        )
    sine = math.sin(angle)
    if abs(sine) < 1.0e-7:
        diagonal = np.maximum(np.diag(rotation) + 1.0, 0.0)
        axis = np.sqrt(diagonal / 2.0)
        if axis[0] > 1.0e-6:
            axis[1] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[0])
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[0])
        elif axis[1] > 1.0e-6:
            axis[2] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[1])
        else:
            axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        return angle * axis
    skew = np.asarray(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    return angle / (2.0 * sine) * skew


class ArmQPSolver:
    """Solve a bounded seven-variable Cartesian velocity QP."""

    def __init__(
        self,
        model: Any,
        data: Any | None = None,
        *,
        side: str | None = None,
        site_name: str | None = None,
        joint_ids: Iterable[int] | None = None,
        qpos_indices: Iterable[int] | None = None,
        dof_indices: Iterable[int] | None = None,
        position_limits: Iterable[Iterable[float]] | None = None,
        velocity_limits: float | Iterable[float] = 2.0,
        home: Iterable[float] | None = None,
        config: QPConfig | None = None,
    ) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required for ArmQPSolver")
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.side = side.lower() if isinstance(side, str) else side
        self.config = config or QPConfig()
        if site_name is None and self.side in {"left", "right"}:
            site_name = f"{self.side[0]}_wrist_target"
        if site_name is None:
            if int(model.nsite) != 1:
                raise ValueError("site_name is required when the model has multiple sites")
            self.site_id = 0
        else:
            self.site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
            if self.site_id < 0:
                raise ValueError(f"unknown site: {site_name}")

        if joint_ids is None:
            joint_ids = self._discover_joint_ids(self.side)
        self.joint_ids = np.asarray(tuple(int(index) for index in joint_ids), dtype=np.int64)
        if self.joint_ids.shape != (7,):
            raise ValueError("ArmQPSolver requires exactly seven hinge joints")
        if np.any(model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("ArmQPSolver joint_ids must reference hinge joints")
        self.qpos_indices = np.asarray(
            tuple(int(value) for value in qpos_indices)
            if qpos_indices is not None
            else tuple(int(model.jnt_qposadr[index]) for index in self.joint_ids),
            dtype=np.int64,
        )
        self.dof_indices = np.asarray(
            tuple(int(value) for value in dof_indices)
            if dof_indices is not None
            else tuple(int(model.jnt_dofadr[index]) for index in self.joint_ids),
            dtype=np.int64,
        )
        if self.qpos_indices.shape != (7,) or self.dof_indices.shape != (7,):
            raise ValueError("qpos_indices and dof_indices must each have seven values")
        if position_limits is None:
            limits = np.asarray(model.jnt_range[self.joint_ids], dtype=np.float64)
            limited = np.asarray(model.jnt_limited[self.joint_ids], dtype=bool)
            limits = np.where(limited[:, None], limits, np.asarray((-math.inf, math.inf)))
        else:
            limits = np.asarray(tuple(position_limits), dtype=np.float64)
        if limits.shape != (7, 2) or np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("position_limits must be seven increasing pairs")
        self.position_limits = limits
        if np.isscalar(velocity_limits):
            self.velocity_limits = np.full(7, float(velocity_limits), dtype=np.float64)
        else:
            self.velocity_limits = np.asarray(tuple(velocity_limits), dtype=np.float64)
        if self.velocity_limits.shape != (7,) or not np.all(np.isfinite(self.velocity_limits)) or np.any(self.velocity_limits <= 0):
            raise ValueError("velocity_limits must contain seven positive finite values")
        self.home = (
            np.asarray(tuple(home), dtype=np.float64)
            if home is not None
            else np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float64).copy()
        )
        if self.home.shape != (7,) or not np.all(np.isfinite(self.home)):
            raise ValueError("home must contain seven finite values")

        # Lower-triangular storage for the symmetric 7×7 Hessian.  OSQP's
        # ``Px`` update then touches exactly the values supplied at setup.
        self._p_rows = np.asarray([row for col in range(7) for row in range(col + 1)], dtype=np.int64)
        self._p_cols = np.asarray([col for col in range(7) for row in range(col + 1)], dtype=np.int64)
        self._p_values = np.zeros(28, dtype=np.float64)
        self._q_values = np.zeros(7, dtype=np.float64)
        self._lower = np.full(7, -1.0, dtype=np.float64)
        self._upper = np.full(7, 1.0, dtype=np.float64)
        self._identity = sparse.eye(7, format="csc")
        initial_p = sparse.csc_matrix((self._p_values.copy(), (self._p_rows, self._p_cols)), shape=(7, 7))
        self._workspace = osqp.OSQP()
        self._workspace.setup(
            P=initial_p,
            q=self._q_values,
            A=self._identity,
            l=self._lower,
            u=self._upper,
            verbose=False,
            polish=False,
            eps_abs=1.0e-8,
            eps_rel=1.0e-8,
            max_iter=4000,
            warm_starting=True,
        )
        self._last_dq = np.zeros(7, dtype=np.float64)
        self._last_q: np.ndarray | None = None

    def _discover_joint_ids(self, side: str | None) -> tuple[int, ...]:
        ids: list[int] = []
        for index in range(int(self.model.njnt)):
            if self.model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index) or "").lower()
            if side == "left" and not (name.startswith(("l_", "left")) or name.endswith(("_l", "_left")) or "_left" in name):
                continue
            if side == "right" and not (name.startswith(("r_", "right")) or name.endswith(("_r", "_right")) or "_right" in name):
                continue
            ids.append(index)
        if len(ids) != 7 and side is not None:
            raise ValueError(f"authoritative manifest must bind exactly seven {side} joints")
        return tuple(ids)

    @property
    def workspace(self) -> Any:
        return self._workspace

    @property
    def last_q(self) -> np.ndarray | None:
        return None if self._last_q is None else self._last_q.copy()

    def _failure(self, status: str, position_error: float = math.inf, orientation_error: float = math.inf) -> ArmSolveResult:
        return ArmSolveResult(np.zeros(7), False, status, position_error, orientation_error)

    def solve(self, q: Iterable[float], target_pose: Any, dt: float) -> ArmSolveResult:
        try:
            q_value = np.asarray(tuple(q), dtype=np.float64)
            if q_value.shape != (7,) or not np.all(np.isfinite(q_value)):
                return self._failure("invalid q")
            if not math.isfinite(float(dt)) or float(dt) <= 0:
                return self._failure("invalid dt")
            target = _pose_matrix(target_pose)
            self.data.qpos[self.qpos_indices] = q_value
            mujoco.mj_forward(self.model, self.data)
            current_position = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64)
            current_rotation = np.asarray(self.data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
            position_error_vector = target[:3, 3] - current_position
            orientation_error_vector = _rotation_log(target[:3, :3] @ current_rotation.T)
            position_error = float(np.linalg.norm(position_error_vector))
            orientation_error = float(np.linalg.norm(orientation_error_vector))
            if position_error > self.config.max_position_error_m or orientation_error > self.config.max_orientation_error_rad:
                return self._failure("target unreachable", position_error, orientation_error)
            desired = np.concatenate((position_error_vector / float(dt), orientation_error_vector / float(dt)))
            desired[:3] = np.clip(desired[:3], -self.config.max_linear_speed, self.config.max_linear_speed)
            desired[3:] = np.clip(desired[3:], -self.config.max_angular_speed, self.config.max_angular_speed)
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jacobian = np.vstack((jacp[:, self.dof_indices], jacr[:, self.dof_indices]))
            weights = np.asarray((self.config.position_weight,) * 3 + (self.config.orientation_weight,) * 3, dtype=np.float64)
            weighted = weights[:, None] * jacobian
            hessian = jacobian.T @ weighted
            hessian.flat[::8] += self.config.lambda_damp + self.config.lambda_home
            gradient = -(jacobian.T @ (weights * desired))
            dq_home = -self.config.home_gain * (q_value - self.home)
            gradient -= self.config.lambda_home * dq_home
            lower = np.maximum(-self.velocity_limits, (self.position_limits[:, 0] - q_value) / float(dt))
            upper = np.minimum(self.velocity_limits, (self.position_limits[:, 1] - q_value) / float(dt))
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
                return self._failure("invalid bounds", position_error, orientation_error)
            self._p_values[:] = hessian[self._p_rows, self._p_cols]
            self._q_values[:] = gradient
            self._lower[:] = lower
            self._upper[:] = upper
            self._workspace.update(Px=self._p_values, q=self._q_values, l=self._lower, u=self._upper)
            self._workspace.warm_start(x=self._last_dq)
            result = self._workspace.solve()
            status = str(result.info.status)
            if status.lower() not in {"solved", "solved inaccurate"}:
                return self._failure(status, position_error, orientation_error)
            dq = np.asarray(result.x, dtype=np.float64)
            if dq.shape != (7,) or not np.all(np.isfinite(dq)) or np.any(dq < lower - 1e-7) or np.any(dq > upper + 1e-7):
                return self._failure("invalid solution", position_error, orientation_error)
            self._last_dq[:] = dq
            self._last_q = q_value + dq * float(dt)
            return ArmSolveResult(dq, True, status, position_error, orientation_error)
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return self._failure(str(exc))

    def reset(self) -> None:
        self._last_dq.fill(0.0)
        self._last_q = None


__all__ = ["ArmQPSolver", "ArmSolveResult", "QPConfig"]
