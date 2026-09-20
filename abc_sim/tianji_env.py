"""CPU MuJoCo rollout for the canonical 54-DoF Tianji/Wuji2 robot.

The scene and its assets are supplied externally. Actions are position-servo
commands, never qpos writes. Success is a simulation-specific hammer lift with
sustained hand contact, not a claim about real-world task completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from abc_sim.rendering.live import create_live_camera_provider


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
CANONICAL_JOINTS = LEFT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS
CAMERA_NAMES = ("top", "left_wrist", "right_wrist")
_BAD_WARNINGS = ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL", "INERTIA", "CONTACTFULL", "CNSTRFULL")


def _vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (54,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [54] vector in radians")
    return array


def _urdf_spec(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    joints = {}
    for node in ET.parse(path).getroot().findall("joint"):
        name = node.get("name")
        if name in joints:
            raise ValueError(f"duplicate URDF joint {name}")
        joints[name] = node
    limits = []
    for name in CANONICAL_JOINTS:
        joint = joints.get(name)
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"URDF must contain revolute joint {name}")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name} has no limit")
        try:
            limits.append([float(limit.attrib[key]) for key in ("lower", "upper", "velocity")])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"URDF joint {name} needs finite lower/upper/velocity limits") from exc
    lower, upper, velocity = np.asarray(limits, dtype=np.float64).T.copy()
    if not np.isfinite(limits).all() or np.any(lower >= upper) or np.any(velocity <= 0):
        raise ValueError("URDF needs finite ordered position limits and positive velocity limits")
    # Each finger's first joint attaches to the palm/wrist body. Starting the
    # contact subtree there includes fixed palm/tip geoms but excludes the arm.
    roots = []
    for hand in (LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS):
        parents = []
        for name in hand[::4]:
            parent = joints[name].find("parent")
            if parent is None or not parent.get("link"):
                raise ValueError(f"URDF joint {name} needs its palm parent link")
            parents.append(parent.attrib["link"])
        if len(set(parents)) != 1:
            raise ValueError("the five finger roots must share a palm/wrist parent")
        roots.append(parents[0])
    if roots[0] == roots[1]:
        raise ValueError("left and right hands must have distinct palm/wrist roots")
    return lower, upper, velocity, tuple(roots)


class TianjiTaskEnv:
    """One fixed-scene, 30 Hz Tianji world with measured-state feedback.

    ``reset`` accepts no scene overrides: use constructor ``initial_qpos`` and
    a prepared MJCF. ``previous_actions`` is the previous tick's measured qpos,
    whereas velocity limiting is relative to the previous applied servo target.
    Rendering is lazy and cached per completed control tick.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        urdf_path: str | Path,
        height: int,
        width: int,
        camera_keys: tuple[str, ...] = CAMERA_NAMES,
        active_cameras: tuple[str, ...] = ("top", "left_wrist"),
        initial_qpos: object | None = None,
        object_body: str = "hammer",
        lift_height: float = 0.05,
        hold_steps: int = 6,
        image_stride: int = 8,
        control_hz: float = 30,
    ):
        if tuple(camera_keys) != CAMERA_NAMES:
            raise ValueError(f"camera_keys must be the canonical order {CAMERA_NAMES}")
        if not active_cameras or len(set(active_cameras)) != len(active_cameras) or not set(active_cameras) <= set(CAMERA_NAMES):
            raise ValueError("active_cameras must be a nonempty unique subset of the canonical cameras")
        for name, value in (("height", height), ("width", width), ("hold_steps", hold_steps), ("image_stride", image_stride)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(lift_height) or lift_height <= 0:
            raise ValueError("lift_height must be finite and positive")
        if control_hz != 30:
            raise ValueError("Tianji policy control_hz must be 30")
        self.height, self.width = int(height), int(width)
        self.camera_keys = CAMERA_NAMES
        self.active_cameras = tuple(active_cameras)
        self.camera_validity = np.array([name in active_cameras for name in CAMERA_NAMES], dtype=bool)
        self.lift_height = float(lift_height)
        self.hold_steps, self.image_stride = int(hold_steps), int(image_stride)
        self.control_hz = float(control_hz)
        self.lower, self.upper, self.velocity_limits, hand_roots = _urdf_spec(urdf_path)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        timestep = float(self.model.opt.timestep)
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("MuJoCo timestep must be finite and positive")
        substeps = 1.0 / (self.control_hz * timestep)
        if not np.isfinite(substeps) or substeps < 1 or not np.isclose(substeps, round(substeps), rtol=0, atol=1e-9):
            raise ValueError("MuJoCo timestep must divide the 1/30 second control period exactly")
        self.control_decimation = int(round(substeps))
        self.joint_names = CANONICAL_JOINTS
        self.joint_ids, self.qpos_indices, self.dof_indices, self.actuator_indices = self._resolve_robot()
        for camera in CAMERA_NAMES:
            self._named_id(mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.object_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, object_body)
        object_joints = np.flatnonzero(self.model.jnt_bodyid == self.object_body_id)
        if len(object_joints) != 1 or self.model.jnt_type[object_joints[0]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"object body {object_body!r} must own one free joint")
        object_bodies = self._descendants(self.object_body_id)
        hand_bodies = np.zeros(self.model.nbody, dtype=bool)
        for root, hand in zip(hand_roots, (LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS), strict=True):
            subtree = self._descendants(self._named_id(mujoco.mjtObj.mjOBJ_BODY, root))
            for name in hand:
                joint = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                if not subtree[self.model.jnt_bodyid[joint]]:
                    raise ValueError(f"hand joint {name} must descend from URDF palm body {root}")
            hand_bodies |= subtree
        arm_ids = [self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS]
        if np.any(hand_bodies[self.model.jnt_bodyid[arm_ids]]) or np.any(hand_bodies & object_bodies):
            raise ValueError("hand contact subtrees must not contain arms or the free object")
        self._object_geoms = object_bodies[self.model.geom_bodyid]
        self._hand_geoms = hand_bodies[self.model.geom_bodyid]
        if not self._object_geoms.any() or not self._hand_geoms.any():
            raise ValueError("object and hand subtrees must contain geoms")
        initial = self.model.qpos0[self.qpos_indices] if initial_qpos is None else initial_qpos
        self.initial_qpos = _vector(initial, "initial_qpos").copy()
        if np.any(self.initial_qpos < self.lower) or np.any(self.initial_qpos > self.upper):
            raise ValueError("initial_qpos must lie strictly within the inclusive URDF joint bounds")
        self._velocity_step = self.velocity_limits / self.control_hz
        self._warning_ids = tuple(int(getattr(mujoco.mjtWarning, f"mjWARN_{name}")) for name in _BAD_WARNINGS)
        self._camera_provider = None
        self._frames: dict[str, np.ndarray] = {}
        self._frame_step = -1
        self._has_reset = False
        self._closed = False
        self.randomization: dict[str, Any] | None = None

    def _named_id(self, kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"MuJoCo scene is missing {kind.name}: {name}")
        return index

    def _resolve_robot(self) -> tuple[np.ndarray, ...]:
        joints, qpos, dofs, actuators = [], [], [], []
        joint_transmission = np.isin(self.model.actuator_trntype, [mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT])
        for index, name in enumerate(CANONICAL_JOINTS):
            joint = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not self.model.jnt_limited[joint]:
                raise ValueError(f"joint {name} must be a limited hinge")
            bounds = np.array([self.lower[index], self.upper[index]])
            if not np.allclose(self.model.jnt_range[joint], bounds, rtol=0, atol=1e-6):
                raise ValueError(f"joint {name} range disagrees with the URDF")
            matches = np.flatnonzero(joint_transmission & (self.model.actuator_trnid[:, 0] == joint))
            if len(matches) != 1:
                raise ValueError(f"joint {name} must have exactly one position actuator")
            actuator = int(matches[0])
            gain = self.model.actuator_gainprm[actuator]
            bias = self.model.actuator_biasprm[actuator]
            if (
                self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                or self.model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                or self.model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                or self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                or not np.isfinite(gain).all() or not np.isfinite(bias).all()
                or gain[0] <= 0 or np.any(gain[1:] != 0)
                or bias[0] != 0 or not np.isclose(bias[1], -gain[0], rtol=1e-10, atol=1e-10)
                or bias[2] > 0 or np.any(bias[3:] != 0)
            ):
                raise ValueError(f"joint {name} needs a stateless position servo (optional nonnegative damping)")
            if not np.array_equal(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
                raise ValueError(f"position actuator for {name} must have unit joint gear")
            if not self.model.actuator_ctrllimited[actuator] or not np.allclose(self.model.actuator_ctrlrange[actuator], bounds, rtol=0, atol=1e-6):
                raise ValueError(f"position actuator for {name} needs the URDF control range")
            joints.append(joint)
            qpos.append(int(self.model.jnt_qposadr[joint]))
            dofs.append(int(self.model.jnt_dofadr[joint]))
            actuators.append(actuator)
        addresses = tuple(np.asarray(values, dtype=np.int64) for values in (joints, qpos, dofs, actuators))
        if any(len(np.unique(values)) != 54 for values in addresses):
            raise ValueError("robot addresses must be 54 unique joint/qpos/dof/actuator indices")
        return addresses

    def _descendants(self, root: int) -> np.ndarray:
        selected = np.zeros(self.model.nbody, dtype=bool)
        selected[root] = True
        # MuJoCo stores parents before their descendants.
        for body in range(root + 1, self.model.nbody):
            selected[body] = selected[self.model.body_parentid[body]]
        return selected

    def _require_reset(self) -> None:
        if self._closed:
            raise RuntimeError("TianjiTaskEnv is closed")
        if not self._has_reset:
            raise RuntimeError("reset must be called before using TianjiTaskEnv")
        if self._fault is not None:
            raise RuntimeError(f"MuJoCo physics failed; reset required: {self._fault}")

    def _check_physics(self, expected_time: float) -> None:
        problem = None
        warnings = [name for name, index in zip(_BAD_WARNINGS, self._warning_ids, strict=True)
                    if self.data.warning[index].number]
        if warnings:
            problem = f"MuJoCo physics warning: {', '.join(warnings)}"
        elif not all(np.isfinite(array).all() for array in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
            problem = "nonfinite MuJoCo position, velocity, acceleration, or control"
        elif not np.isfinite(self.data.time) or not np.isclose(self.data.time, expected_time, rtol=1e-10, atol=1e-10):
            problem = f"MuJoCo time anomaly: expected {expected_time}, got {self.data.time}"
        if problem is not None:
            self._fault = problem
            raise RuntimeError(problem)

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("TianjiTaskEnv is closed")
        if options is not None and (not isinstance(options, dict) or options):
            raise ValueError("Tianji reset accepts only None or {}; the prepared scene is fixed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.initial_qpos
        self.data.ctrl[self.actuator_indices] = self.initial_qpos
        self._previous_measured = self.initial_qpos.copy()
        self.applied_target = self.initial_qpos.copy()
        self._step = 0
        self._fault = None
        self._has_reset = True
        self._frames = {}
        self._frame_step = -1
        if self._camera_provider is not None:
            self._camera_provider.reset()
        self.randomization = {"mode": "fixed_scene", "seed": int(seed), "randomized": False}
        self._bound_clips = self._velocity_clips = self._total_clips = 0
        self._last_bound_clips = self._last_velocity_clips = self._last_clips = 0
        self._tracking_squared_sum = 0.0
        self._tracking_rmse = 0.0
        self._hold_count = 0
        self._ever_success = False
        mujoco.mj_forward(self.model, self.data)
        self._check_physics(0.0)
        self._initial_object_height = float(self.data.xpos[self.object_body_id, 2])
        self._update_task_metrics(advance=False)
        return self.obs()

    def obs(self) -> dict[str, Any]:
        self._require_reset()
        self._check_physics(self._step / self.control_hz)
        frames = self.render_cameras() if self._step % self.image_stride == 0 else {}
        return {
            "state": self.data.qpos[self.qpos_indices].astype(np.float32),
            "previous_actions": self._previous_measured.astype(np.float32),
            "images": {name: frames[name] for name in self.active_cameras} if frames else {},
            "camera_validity": self.camera_validity.copy(),
            "step": self._step,
        }

    def step_one(self, action: np.ndarray) -> None:
        self._require_reset()
        command = _vector(action, "action")
        self._check_physics(self._step / self.control_hz)
        bounded = np.clip(command, self.lower, self.upper)
        target = np.clip(bounded, self.applied_target - self._velocity_step, self.applied_target + self._velocity_step)
        self._last_bound_clips = int(np.count_nonzero(command != bounded))
        self._last_velocity_clips = int(np.count_nonzero(bounded != target))
        self._last_clips = int(np.count_nonzero(command != target))
        self._bound_clips += self._last_bound_clips
        self._velocity_clips += self._last_velocity_clips
        self._total_clips += self._last_clips
        self._previous_measured = self.data.qpos[self.qpos_indices].copy()
        self.applied_target = target
        self.data.ctrl[self.actuator_indices] = target
        start_time = self._step / self.control_hz
        for substep in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            self._check_physics(start_time + (substep + 1) * self.model.opt.timestep)
        # mj_step integrates qpos after computing kinematics. Refresh contacts
        # and body poses so metrics and rendered frames describe the new state.
        mujoco.mj_forward(self.model, self.data)
        self._check_physics((self._step + 1) / self.control_hz)
        self._step += 1
        self._frames = {}
        self._frame_step = -1
        error = self.data.qpos[self.qpos_indices] - target
        squared_error = float(np.dot(error, error))
        self._tracking_rmse = float(np.sqrt(squared_error / 54))
        self._tracking_squared_sum += squared_error
        self._update_task_metrics(advance=True)

    def _update_task_metrics(self, *, advance: bool) -> None:
        self._height_gain = float(self.data.xpos[self.object_body_id, 2]) - self._initial_object_height
        self._hand_contact = False
        for contact in self.data.contact[:self.data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            # Positive-margin, inactive proximity contacts are not grasps.
            if first < 0 or second < 0 or contact.dist > 0 or contact.efc_address < 0:
                continue
            if (self._object_geoms[first] and self._hand_geoms[second]) or (self._object_geoms[second] and self._hand_geoms[first]):
                self._hand_contact = True
                break
        lifted_contact = self._height_gain >= self.lift_height and self._hand_contact
        if advance:
            self._hold_count = self._hold_count + 1 if lifted_contact else 0
        self._success = lifted_contact and self._hold_count >= self.hold_steps
        self._ever_success = self._ever_success or self._success

    def evaluate(self) -> dict[str, Any]:
        """Return the completed tick's metrics without advancing the hold timer."""
        self._require_reset()
        self._check_physics(self._step / self.control_hz)
        return {
            "success": bool(self._success),
            "ever_success": bool(self._ever_success),
            "reward": float(np.clip(self._height_gain / self.lift_height, 0, 1)),
            "object_height_gain": self._height_gain,
            "hand_object_contact": self._hand_contact,
            "hold_count": self._hold_count,
            "tracking_rmse": self._tracking_rmse,
            "tracking_rmse_episode": float(np.sqrt(self._tracking_squared_sum / (54 * self._step))) if self._step else 0.0,
            "joint_bound_clips": self._last_bound_clips,
            "velocity_limit_clips": self._last_velocity_clips,
            "limit_clips": self._last_clips,
            "joint_bound_clips_total": self._bound_clips,
            "velocity_limit_clips_total": self._velocity_clips,
            "limit_clips_total": self._total_clips,
            "limit_clip_fraction": self._total_clips / (54 * self._step) if self._step else 0.0,
            "physics_ok": True,
            "physics_backend": "mujoco_cpu",
            "step": self._step,
            "sim_time": float(self.data.time),
            "mujoco_warnings": {name: int(self.data.warning[index].number) for name, index in zip(_BAD_WARNINGS, self._warning_ids, strict=True)},
        }

    def render_cameras(self) -> dict[str, np.ndarray]:
        self._require_reset()
        self._check_physics(self._step / self.control_hz)
        if self._frame_step != self._step:
            if self._camera_provider is None:
                self._camera_provider = create_live_camera_provider(
                    model=self.model, data=self.data, backend="mujoco",
                    width=self.width, height=self.height, camera_names=self.camera_keys,
                )
            frames = self._camera_provider.frames_for_step(self._step, float(self.data.time))
            converted = {}
            for name in self.camera_keys:
                image = np.asarray(frames[name])
                if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
                    raise RuntimeError(f"camera {name} must return HWC uint8 RGB frames")
                converted[name] = np.ascontiguousarray(image.transpose(2, 0, 1))
            self._frames = converted
            self._frame_step = self._step
        return dict(self._frames)

    def close(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.close()
            self._camera_provider = None
        self._frames = {}
        self._closed = True

    step_one_vanilla = step_one
    evaluate_vanilla = evaluate
    obs_vanilla_state = obs
    render_cameras_vanilla_state = render_cameras
