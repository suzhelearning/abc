"""Replay and verify the state stream of a SPD-VR HDF5 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .data import validate_episode
from .robot import RobotSpec
from .simulation import SPDVRSim


class ReplayError(ValueError):
    """Raised when an episode cannot be reproduced by the supplied model."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_episode(
    path: str | Path,
    *,
    simulator: Any | None = None,
    model_path: str | Path | None = None,
    start: int = 0,
    stop: int | None = None,
    tolerance: float = 1e-6,
    render: bool = False,
) -> dict[str, Any]:
    """Restore every recorded full state and compare the derived robot state.

    The episode's raw stream remains the source of truth.  No controls are
    replayed here: restoring ``mjSTATE_FULLPHYSICS`` is deterministic and also
    covers scene free-joints, actuators, and contact-relevant state.  Rendering
    is optional and only checks that all policy camera outputs can be produced.
    """
    episode_path = Path(path)
    manifest = validate_episode(episode_path, verify_checksums=True)
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ReplayError("tolerance must be finite and non-negative")
    if model_path is not None:
        model = Path(model_path)
        if not model.is_file():
            raise ReplayError(f"replay model does not exist: {model}")
        expected = manifest.get("model_sha256")
        actual = _sha256(model)
        if expected is not None and actual != expected:
            raise ReplayError(
                f"model sha256 differs from episode manifest: expected {expected}, got {actual}"
            )
    with h5py.File(episode_path, "r") as handle:
        timestamps = handle["raw/timestamp_ns"][:]
        expected_qpos = handle["raw/observation/qpos"][:]
        expected_qvel = handle["raw/observation/qvel"][:]
        expected_state = handle["raw/mujoco/full_state"][:]
        expected_contact = handle["raw/contacts/hand_object"][:]
        length = int(timestamps.size)
        first = int(start)
        last = length if stop is None else int(stop)
        if first < 0 or last < first or last > length:
            raise ReplayError(f"replay range must satisfy 0 <= start <= stop <= {length}")
        if simulator is None:
            return {
                "valid": True,
                "episode": str(episode_path),
                "raw_frames": length,
                "replayed_frames": 0,
                "replayed": False,
                "range": [first, last],
            }
        if not hasattr(simulator, "restore_full_state"):
            raise ReplayError("simulator must expose restore_full_state")
        max_state_error = 0.0
        max_qpos_error = 0.0
        max_qvel_error = 0.0
        contact_mismatches = 0
        rendered_frames = 0
        for index in range(first, last):
            simulator.restore_full_state(expected_state[index])
            actual_state = np.asarray(simulator.full_state(), dtype=np.float64)
            actual_qpos = simulator.addresses.read_qpos(simulator.data)
            actual_qvel = simulator.addresses.read_qvel(simulator.data)
            state_error = float(np.max(np.abs(actual_state - expected_state[index])))
            qpos_error = float(np.max(np.abs(actual_qpos - expected_qpos[index])))
            qvel_error = float(np.max(np.abs(actual_qvel - expected_qvel[index])))
            max_state_error = max(max_state_error, state_error)
            max_qpos_error = max(max_qpos_error, qpos_error)
            max_qvel_error = max(max_qvel_error, qvel_error)
            if max(state_error, qpos_error, qvel_error) > tolerance:
                raise ReplayError(
                    f"replay mismatch at frame {index}: "
                    f"state={state_error:g}, qpos={qpos_error:g}, qvel={qvel_error:g}"
                )
            object_ids = getattr(simulator, "_object_body_ids", ())
            if object_ids:
                actual_contact = bool(simulator.contact_state().get("hand_object", False))
                if actual_contact != bool(expected_contact[index]):
                    contact_mismatches += 1
            if render:
                frames = simulator.render_cameras()
                if not frames:
                    raise ReplayError(f"no camera frames at replay index {index}")
                rendered_frames += 1
        return {
            "valid": contact_mismatches == 0,
            "episode": str(episode_path),
            "raw_frames": length,
            "replayed_frames": last - first,
            "replayed": True,
            "range": [first, last],
            "max_full_state_error": max_state_error,
            "max_qpos_error": max_qpos_error,
            "max_qvel_error": max_qvel_error,
            "contact_mismatches": contact_mismatches,
            "rendered_frames": rendered_frames,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--object-body", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    robot = RobotSpec.from_urdf(args.urdf)
    with SPDVRSim(
        args.model,
        robot,
        backend="mujoco",
        object_bodies=args.object_body,
    ) as simulator:
        report = replay_episode(
            args.episode,
            simulator=simulator,
            model_path=args.model,
            start=args.start,
            stop=args.stop,
            tolerance=args.tolerance,
            render=args.render,
        )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0 if report["valid"] else 2


__all__ = ["ReplayError", "main", "replay_episode"]


if __name__ == "__main__":
    raise SystemExit(main())
