"""Tianji-Wuji2 simulation adapter built on ABC's MuJoCo/MJWarp backend."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CAMERA_NAMES, JointTarget
from .data import CameraFrame, IMAGE_HEIGHT, IMAGE_WIDTH, RawFrame
from .robot import RobotSpec


class SPDVRSim:
    """One authoritative MuJoCo plant; it can step with MuJoCo or ABC MJWarp."""

    def __init__(
        self,
        model_path: str | Path,
        robot: RobotSpec,
        *,
        backend: str = "mujoco",
        physics_hz: int = 480,
        control_hz: int = 60,
        gpu_id: int | None = None,
        object_bodies: Sequence[str] = (),
    ) -> None:
        if backend not in {"mujoco", "mjwarp"}:
            raise ValueError("backend must be mujoco or mjwarp")
        if physics_hz <= 0 or control_hz <= 0:
            raise ValueError("physics_hz and control_hz must be positive")
        if physics_hz % control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - dependency setup
            raise RuntimeError("MuJoCo is required for SPD-VR simulation") from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        # Scene resets may randomize object mass.  Keep the compiler-provided
        # inertial tensor so the randomized mass can scale inertia coherently
        # instead of leaving MuJoCo's inverse-mass constants stale.
        self._base_body_mass = np.asarray(self.model.body_mass, dtype=np.float64).copy()
        self._base_body_inertia = np.asarray(self.model.body_inertia, dtype=np.float64).copy()
        self.model.opt.timestep = 1.0 / physics_hz
        self.robot = robot
        self.addresses = robot.resolve_mujoco(self.model)
        self.backend = backend
        self.physics_hz = int(physics_hz)
        self.control_hz = int(control_hz)
        self.decimation = physics_hz // control_hz
        self.paused = False
        if backend == "mjwarp":
            # Import lazily: vanilla MuJoCo collection must not require Torch/Warp.
            from abc_minimal.eval_policy import MJWarpSim

            self._warp = MJWarpSim(
                self.model,
                self.data,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                gpu_id=gpu_id,
            )
        else:
            self._warp = None
        self._renderer = None
        self._target = self.addresses.read_qpos(self.data)
        camera_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(self.model.ncam)
        }
        missing = set(CAMERA_NAMES) - camera_names
        if missing:
            raise ValueError(f"simulation model is missing policy cameras: {sorted(missing)}")
        self._set_object_bodies(object_bodies)
        self._hand_body_ids = frozenset(
            index
            for index in range(self.model.nbody)
            if any(
                token in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index) or "").lower()
                for token in ("hand", "palm", "finger", "thumb", "wrist")
            )
        )

    def _has_ancestor(self, body_id: int, ancestors: frozenset[int]) -> bool:
        while body_id:
            if body_id in ancestors:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return body_id in ancestors

    def _set_object_bodies(self, object_bodies: Sequence[str]) -> None:
        ids = []
        for name in object_bodies:
            body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, str(name)
            )
            if body_id < 0:
                raise ValueError(f"simulation model has no object body {name!r}")
            if body_id == 0:
                raise ValueError("the MuJoCo world body cannot be a task object")
            ids.append(int(body_id))
        if len(set(ids)) != len(ids):
            raise ValueError("object body names must be unique")
        self._object_body_ids = tuple(ids)
        object_roots = frozenset(self._object_body_ids)
        self._object_contact_body_ids = frozenset(
            body_id
            for body_id in range(self.model.nbody)
            if self._has_ancestor(body_id, object_roots)
        )

    def reset_scene(self, scene: Any | Mapping[str, Any]) -> None:
        """Reset free task bodies and physical appearance from a scene manifest."""
        manifest = scene.manifest() if hasattr(scene, "manifest") else scene
        if not isinstance(manifest, Mapping):
            raise ValueError("scene must provide a mapping manifest")
        if "objects" not in manifest and isinstance(manifest.get("reset"), Mapping):
            manifest = manifest["reset"]
        objects = manifest.get("objects")
        if not isinstance(objects, list):
            raise ValueError("scene manifest must contain an objects list")
        names = [item.get("name") for item in objects if isinstance(item, Mapping)]
        if len(names) != len(objects) or not all(isinstance(name, str) for name in names):
            raise ValueError("scene objects must contain string names")
        self._set_object_bodies(names)
        self.mujoco.mj_resetData(self.model, self.data)
        for item in objects:
            if not isinstance(item, Mapping):
                raise ValueError("scene object entry must be a mapping")
            name = str(item["name"])
            body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, name
            )
            position = np.asarray(item.get("position"), dtype=np.float64)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"scene object {name!r} has invalid position")
            yaw = float(item.get("yaw_rad", 0.0))
            if not math.isfinite(yaw):
                raise ValueError(f"scene object {name!r} has invalid yaw")
            joint_start = int(self.model.body_jntadr[body_id])
            joint_count = int(self.model.body_jntnum[body_id])
            free_joint = None
            for joint_id in range(joint_start, joint_start + joint_count):
                if self.model.jnt_type[joint_id] == self.mujoco.mjtJoint.mjJNT_FREE:
                    free_joint = joint_id
                    break
            if free_joint is None:
                raise ValueError(f"scene object {name!r} must have a free joint")
            qpos_start = int(self.model.jnt_qposadr[free_joint])
            self.data.qpos[qpos_start : qpos_start + 7] = (
                *position,
                math.cos(yaw * 0.5),
                0.0,
                0.0,
                math.sin(yaw * 0.5),
            )
            mass = float(item.get("mass_kg"))
            friction = float(item.get("friction"))
            if not math.isfinite(mass) or mass <= 0 or not math.isfinite(friction) or friction <= 0:
                raise ValueError(f"scene object {name!r} has invalid physical parameters")
            base_mass = float(self._base_body_mass[body_id])
            if not math.isfinite(base_mass) or base_mass <= 0:
                raise ValueError(f"scene object {name!r} has no positive compiled mass")
            self.model.body_mass[body_id] = mass
            self.model.body_inertia[body_id] = self._base_body_inertia[body_id] * (mass / base_mass)
            for geom in item.get("geoms", ()):
                if not isinstance(geom, Mapping) or not isinstance(geom.get("name"), str):
                    raise ValueError(f"scene object {name!r} has invalid geom metadata")
                geom_id = self.mujoco.mj_name2id(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom["name"]
                )
                if geom_id < 0:
                    raise ValueError(f"scene object {name!r} geom is missing: {geom['name']}")
                self.model.geom_friction[geom_id, 0] = friction
                rgba = np.asarray(geom.get("rgba"), dtype=np.float64)
                if rgba.shape != (4,) or not np.all(np.isfinite(rgba)):
                    raise ValueError(f"scene object {name!r} has invalid rgba")
                self.model.geom_rgba[geom_id] = rgba
        self.mujoco.mj_setConst(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self._target = self.addresses.read_qpos(self.data)
        self.addresses.write_control(self.data, self._target)
        if self._warp is not None:
            self._warp.load_state()

    def close(self) -> None:
        if self._warp is not None:
            self._warp.close()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def set_paused(self, paused: bool) -> None:
        """Freeze or resume stepping without changing the last safe target."""

        self.paused = bool(paused)

    def reset(self) -> None:
        self.paused = False
        self.mujoco.mj_resetData(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self._target = self.addresses.read_qpos(self.data)
        self.addresses.write_control(self.data, self._target)
        if self._warp is not None:
            self._warp.load_state()

    def set_target(self, target: JointTarget | np.ndarray) -> None:
        value = target.qpos if isinstance(target, JointTarget) else target
        self._target = self.robot.clip(value)
        if self._warp is None:
            self.addresses.write_control(self.data, self._target)
        else:
            control = np.asarray(self.data.ctrl, dtype=np.float32).copy()
            control[self.addresses.actuator] = self._target
            self._warp.set_ctrl(control)

    def step(self) -> None:
        if self.paused:
            return
        if self._warp is None:
            self.mujoco.mj_step(self.model, self.data, nstep=self.decimation)
        else:
            self._warp.step(self.decimation)

    def benchmark(self, duration_s: float, *, render: bool = False) -> dict[str, float | int | bool]:
        """Measure the control loop without starting a viewer or recorder.

        ``duration_s`` is measured in 60 Hz control ticks; each tick advances
        the authoritative 480 Hz physics model by ``decimation`` steps.  The
        result is a diagnostic only and never qualifies a contact model or a
        policy for deployment.
        """
        if not math.isfinite(float(duration_s)) or float(duration_s) <= 0.0:
            raise ValueError("duration_s must be positive")
        ticks = max(1, int(round(float(duration_s) * self.control_hz)))
        samples: list[float] = []
        rendered = 0
        for _ in range(ticks):
            before = time.perf_counter_ns()
            self.step()
            if render:
                self.render_cameras()
                rendered += 1
            samples.append((time.perf_counter_ns() - before) / 1e6)
        values = np.asarray(samples, dtype=np.float64)
        return {
            "duration_s": float(duration_s),
            "control_ticks": ticks,
            "physics_steps": ticks * self.decimation,
            "control_hz": self.control_hz,
            "physics_hz": self.control_hz * self.decimation,
            "render": bool(render),
            "rendered_ticks": rendered,
            "step_p50_ms": float(np.percentile(values, 50)),
            "step_p95_ms": float(np.percentile(values, 95)),
            "step_p99_ms": float(np.percentile(values, 99)),
            "step_max_ms": float(np.max(values)),
            "realtime_budget_ms": 1000.0 / self.control_hz,
            "realtime_p95_ok": bool(float(np.percentile(values, 95)) <= 1000.0 / self.control_hz),
        }

    def _sync_from_warp(self) -> None:
        if self._warp is None:
            return
        self.data.qpos[:] = self._warp.d_warp.qpos.numpy()[0]
        self.data.qvel[:] = self._warp.d_warp.qvel.numpy()[0]
        self.data.ctrl[:] = self._warp.d_warp.ctrl.numpy()[0]
        if self.model.na and hasattr(self._warp.d_warp, "act"):
            self.data.act[:] = self._warp.d_warp.act.numpy()[0]
        if hasattr(self._warp.d_warp, "time"):
            self.data.time = float(np.asarray(self._warp.d_warp.time.numpy()).reshape(-1)[0])
        self.mujoco.mj_forward(self.model, self.data)

    def sync_for_viewer(self) -> None:
        self._sync_from_warp()

    def full_state(self) -> np.ndarray:
        self._sync_from_warp()
        spec = self.mujoco.mjtState.mjSTATE_FULLPHYSICS
        result = np.empty(self.mujoco.mj_stateSize(self.model, spec), dtype=np.float64)
        self.mujoco.mj_getState(self.model, self.data, result, spec)
        return result

    def restore_full_state(self, state: np.ndarray) -> None:
        spec = self.mujoco.mjtState.mjSTATE_FULLPHYSICS
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (self.mujoco.mj_stateSize(self.model, spec),):
            raise ValueError("full MuJoCo state width does not match the model")
        self.mujoco.mj_setState(self.model, self.data, value, spec)
        self.mujoco.mj_forward(self.model, self.data)
        self._target = np.asarray(self.data.ctrl[self.addresses.actuator], dtype=np.float64).copy()
        if self._warp is not None:
            self._warp.load_state()

    def render_cameras(self) -> dict[str, CameraFrame]:
        self._sync_from_warp()
        if self._renderer is None:
            self._renderer = self.mujoco.Renderer(
                self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
            )
        result = {}
        for camera in CAMERA_NAMES:
            self._renderer.disable_segmentation_rendering()
            self._renderer.update_scene(self.data, camera=camera)
            rgb = self._renderer.render().copy()
            self._renderer.enable_segmentation_rendering()
            self._renderer.update_scene(self.data, camera=camera)
            segmentation = self._renderer.render().astype(np.int32, copy=True)
            result[camera] = CameraFrame(rgb, segmentation)
        self._renderer.disable_segmentation_rendering()
        return result

    def object_state(self) -> list[dict[str, Any]]:
        """Return world-frame state for the task bodies selected by the caller."""
        self._sync_from_warp()
        result = []
        for body_id in self._object_body_ids:
            result.append(
                {
                    "name": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
                    ),
                    "position": self.data.xpos[body_id].tolist(),
                    "quaternion_wxyz": self.data.xquat[body_id].tolist(),
                    "spatial_velocity": self.data.cvel[body_id].tolist(),
                }
            )
        return result

    def contact_state(self) -> dict[str, Any]:
        """Return replay-audit contact records and a hand/object summary bit."""
        self._sync_from_warp()
        records = []
        hand_object = False
        object_ids = self._object_contact_body_ids
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            is_hand_object = (
                body1 in self._hand_body_ids and body2 in object_ids
            ) or (
                body2 in self._hand_body_ids and body1 in object_ids
            )
            hand_object = hand_object or is_hand_object
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, self.data, index, force)
            records.append(
                {
                    "geom1": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom1
                    ),
                    "geom2": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom2
                    ),
                    "body1": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_BODY, body1
                    ),
                    "body2": self.mujoco.mj_id2name(
                        self.model, self.mujoco.mjtObj.mjOBJ_BODY, body2
                    ),
                    "position": np.asarray(contact.pos).tolist(),
                    "frame": np.asarray(contact.frame).tolist(),
                    "distance": float(contact.dist),
                    "force_torque": force.tolist(),
                    "hand_object": is_hand_object,
                }
            )
        return {"hand_object": hand_object, "records": records}

    def has_task_object_contact(self) -> bool:
        """Small predicate used by the checkpoint state machine."""

        return bool(self.contact_state().get("hand_object", False))

    def record_frame(
        self,
        timestamp_ns: int,
        pico_hands: np.ndarray,
        pico_timestamp_ns: int,
        pico_bridge_monotonic_ns: int,
        pico_sequence_id: int,
        tracking_epoch: int,
        pico_scale: np.ndarray,
        validity: np.ndarray,
        *,
        objects: Any = None,
        contacts: Any = None,
    ) -> RawFrame:
        if self._warp is not None:
            raise RuntimeError(
                "HDF5 recording currently requires the mujoco backend so contact "
                "forces and mjSTATE_FULLPHYSICS come from one authoritative data object"
            )
        self._sync_from_warp()
        return RawFrame(
            timestamp_ns=timestamp_ns,
            qpos=self.addresses.read_qpos(self.data),
            qvel=self.addresses.read_qvel(self.data),
            qpos_target=self._target,
            mujoco_full_state=self.full_state(),
            cameras=self.render_cameras(),
            pico_hands=pico_hands,
            pico_timestamp_ns=pico_timestamp_ns,
            pico_bridge_monotonic_ns=pico_bridge_monotonic_ns,
            pico_sequence_id=pico_sequence_id,
            tracking_epoch=tracking_epoch,
            pico_scale=pico_scale,
            validity=validity,
            objects=self.object_state() if objects is None else objects,
            contacts=self.contact_state() if contacts is None else contacts,
        )

    def __enter__(self) -> "SPDVRSim":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--object-body", action="append", default=[])
    args = parser.parse_args(argv)
    robot = RobotSpec.from_urdf(args.urdf)
    with SPDVRSim(
        args.model,
        robot,
        backend=args.backend,
        gpu_id=args.gpu_id,
        object_bodies=args.object_body,
    ) as simulation:
        print(json.dumps(simulation.benchmark(args.duration, render=args.render), sort_keys=True))
    return 0


__all__ = ["SPDVRSim", "benchmark_main"]

if __name__ == "__main__":
    raise SystemExit(benchmark_main())
