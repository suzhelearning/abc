"""Operator-safe episode state machine for SPD-VR data collection.

The state machine is deliberately independent from the transport and recorder
implementations.  It serializes operator commands, refuses checkpoints while
the simulator reports hand/object contact, and restores the complete MuJoCo
state (including the task objects) on revert.  A skipped episode is discarded
without ever publishing a partial HDF5 file.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import queue
from typing import Any, Mapping

import numpy as np


class EpisodeState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    FAULT = "FAULT"


class EpisodeCommandType(str, Enum):
    START = "start"
    CHECKPOINT = "checkpoint"
    PAUSE = "pause"
    RESUME = "resume"
    REVERT = "revert"
    SKIP = "skip"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class EpisodeCommand:
    type: EpisodeCommandType
    reason: str = ""


@dataclass
class _Checkpoint:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray | None
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    time: float
    rng_state: dict[str, Any]
    counters: dict[str, int]
    state_epoch: int
    full_state: np.ndarray | None
    target: np.ndarray | None = None


def _mujoco(simulator: Any) -> Any | None:
    """Return either the public or legacy MuJoCo handle used by a simulator."""

    return getattr(simulator, "mujoco", None) or getattr(simulator, "_mujoco", None)


class EpisodeController:
    """Serialize collection commands through one queue and state transition."""

    def __init__(
        self,
        simulator: Any,
        task_spec: Any,
        *,
        seed: int = 0,
        recorder: Any | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.simulator = simulator
        self.task_spec = task_spec
        self.seed = int(seed)
        self.recorder = recorder
        self._has_run_metadata = run_metadata is not None
        self.run_metadata = copy.deepcopy(dict(run_metadata or {}))
        self.state = EpisodeState.IDLE
        self.state_epoch = 0
        self.episode_id = 0
        self.counters = {"frames": 0, "checkpoints": 0, "reverts": 0, "skips": 0}
        self._commands: queue.SimpleQueue[EpisodeCommand] = queue.SimpleQueue()
        self._checkpoint: _Checkpoint | None = None
        self._rng = np.random.default_rng(self.seed)
        self._manifest: dict[str, Any] | None = None
        self.last_error: str | None = None

    @property
    def manifest(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._manifest)

    def enqueue(
        self, command: EpisodeCommand | EpisodeCommandType | str, reason: str = ""
    ) -> None:
        if not isinstance(command, EpisodeCommand):
            command = EpisodeCommand(EpisodeCommandType(command), reason)
        self._commands.put(command)

    def on_record_flag(self, enabled: bool) -> None:
        self.enqueue(
            EpisodeCommandType.START if enabled else EpisodeCommandType.FINISH,
            "pico_record_flag",
        )

    def _has_task_object_contact(self) -> bool:
        method = getattr(self.simulator, "has_task_object_contact", None)
        if method is not None:
            return bool(method())
        method = getattr(self.simulator, "contact_state", None)
        if method is None:
            return False
        value = method()
        return bool(value.get("hand_object", False)) if isinstance(value, Mapping) else bool(value)

    def _snapshot(self) -> _Checkpoint:
        data = self.simulator.data
        mujoco = _mujoco(self.simulator)
        full_state = None
        if mujoco is not None and hasattr(mujoco, "mj_stateSize"):
            spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
            state_size = mujoco.mj_stateSize(self.simulator.model, spec)
            full_state = np.empty(state_size, dtype=np.float64)
            mujoco.mj_getState(self.simulator.model, data, full_state, spec)
        return _Checkpoint(
            qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(data.qvel, dtype=np.float64).copy(),
            act=(
                np.asarray(data.act, dtype=np.float64).copy()
                if getattr(data, "act", None) is not None
                else None
            ),
            ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
            mocap_pos=np.asarray(getattr(data, "mocap_pos", np.empty((0, 3))), dtype=np.float64).copy(),
            mocap_quat=np.asarray(getattr(data, "mocap_quat", np.empty((0, 4))), dtype=np.float64).copy(),
            time=float(data.time),
            rng_state=copy.deepcopy(self._rng.bit_generator.state),
            counters=dict(self.counters),
            state_epoch=self.state_epoch,
            full_state=full_state,
            target=(
                np.asarray(self.simulator._target, dtype=np.float64).copy()
                if hasattr(self.simulator, "_target")
                else None
            ),
        )

    def _restore(self, checkpoint: _Checkpoint) -> None:
        data = self.simulator.data
        mujoco = _mujoco(self.simulator)
        if checkpoint.full_state is not None and mujoco is not None:
            spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
            mujoco.mj_setState(self.simulator.model, data, checkpoint.full_state, spec)
        data.qpos[:] = checkpoint.qpos
        data.qvel[:] = checkpoint.qvel
        if checkpoint.act is not None and getattr(data, "act", None) is not None:
            data.act[:] = checkpoint.act
        data.ctrl[:] = checkpoint.ctrl
        if getattr(data, "mocap_pos", None) is not None:
            data.mocap_pos[:] = checkpoint.mocap_pos
        if getattr(data, "mocap_quat", None) is not None:
            data.mocap_quat[:] = checkpoint.mocap_quat
        data.time = checkpoint.time
        if hasattr(self.simulator, "tick"):
            self.simulator.tick = int(
                round(float(checkpoint.time) * float(getattr(self.simulator, "physics_hz", 480)))
            )
        if mujoco is not None:
            mujoco.mj_forward(self.simulator.model, data)
        if checkpoint.target is not None and hasattr(self.simulator, "_target"):
            self.simulator._target = checkpoint.target.copy()
        else:
            # Position actuators are the canonical target representation for
            # SPDVRSim.  Keep it synchronized on simulators that expose the
            # address map but predate the explicit checkpoint target field.
            addresses = getattr(self.simulator, "addresses", None)
            actuator = getattr(addresses, "actuator", None)
            if actuator is not None and hasattr(self.simulator, "_target"):
                values = np.asarray(data.ctrl, dtype=np.float64)[actuator]
                if values.shape == np.asarray(self.simulator._target).shape:
                    self.simulator._target = values.copy()
        self._rng.bit_generator.state = copy.deepcopy(checkpoint.rng_state)
        self.counters = dict(checkpoint.counters)
        invalidate = getattr(self.simulator, "invalidate_snapshots", None)
        if invalidate is not None:
            invalidate()
        load_state = getattr(getattr(self.simulator, "_warp", None), "load_state", None)
        if load_state is not None:
            load_state()

    def _set_paused(self, value: bool) -> None:
        setter = getattr(self.simulator, "set_paused", None)
        if setter is not None:
            setter(bool(value))

    def _start(self) -> str:
        if self.state is not EpisodeState.IDLE:
            return f"start ignored in {self.state.value}"
        self.episode_id += 1
        self.state_epoch += 1
        self._set_paused(False)
        scene_result = self.task_spec.reset(self.seed + self.episode_id - 1)
        self._manifest = copy.deepcopy(scene_result.manifest())
        if self._has_run_metadata:
            self._manifest["teleop"] = copy.deepcopy(self.run_metadata)
        reset_scene = getattr(self.simulator, "reset_scene", None)
        if reset_scene is not None:
            reset_scene(scene_result)
        set_objects = getattr(self.simulator, "set_task_object_body_names", None)
        if set_objects is not None:
            set_objects({item.name for item in getattr(scene_result, "objects", ())})
        self._checkpoint = None
        self.state = EpisodeState.RECORDING
        self.last_error = None
        if self.recorder is not None:
            self.recorder.start_episode(self.episode_id, self._manifest)
        return "started"

    def _checkpoint_now(self) -> str:
        if self.state is not EpisodeState.RECORDING:
            return f"checkpoint ignored in {self.state.value}"
        if self._has_task_object_contact():
            return "checkpoint rejected while hand-object contact is active"
        self._checkpoint = self._snapshot()
        self.counters["checkpoints"] += 1
        return "checkpointed"

    def _pause(self) -> str:
        if self.state is not EpisodeState.RECORDING:
            return f"pause ignored in {self.state.value}"
        self._set_paused(True)
        self.state = EpisodeState.PAUSED
        self.state_epoch += 1
        return "paused"

    def _resume(self) -> str:
        if self.state is not EpisodeState.PAUSED:
            return f"resume ignored in {self.state.value}"
        self._set_paused(False)
        self.state = EpisodeState.RECORDING
        self.state_epoch += 1
        return "resumed"

    def _revert(self) -> str:
        if self.state not in {EpisodeState.RECORDING, EpisodeState.PAUSED}:
            return f"revert ignored in {self.state.value}"
        if self._checkpoint is None:
            return "revert rejected: no checkpoint"
        self._restore(self._checkpoint)
        self.state_epoch += 1
        self.counters["reverts"] += 1
        return "reverted"

    def _finish(self) -> str:
        if self.state not in {EpisodeState.RECORDING, EpisodeState.PAUSED}:
            return f"finish ignored in {self.state.value}"
        self._set_paused(False)
        if self.recorder is not None:
            self.recorder.finish_episode()
        self.state = EpisodeState.IDLE
        self.state_epoch += 1
        self._checkpoint = None
        return "finished"

    def _skip(self) -> str:
        if self.state not in {EpisodeState.RECORDING, EpisodeState.PAUSED}:
            return f"skip ignored in {self.state.value}"
        self._set_paused(False)
        if self.recorder is not None:
            self.recorder.discard_episode("operator_skip")
        self.counters["skips"] += 1
        self.state = EpisodeState.IDLE
        self.state_epoch += 1
        self._checkpoint = None
        return "skipped"

    def process_one(self) -> str | None:
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return None
        try:
            handlers = {
                EpisodeCommandType.START: self._start,
                EpisodeCommandType.CHECKPOINT: self._checkpoint_now,
                EpisodeCommandType.PAUSE: self._pause,
                EpisodeCommandType.RESUME: self._resume,
                EpisodeCommandType.REVERT: self._revert,
                EpisodeCommandType.SKIP: self._skip,
                EpisodeCommandType.FINISH: self._finish,
            }
            return handlers[command.type]()
        except Exception as exc:
            self.state = EpisodeState.FAULT
            self.last_error = str(exc)
            return f"fault: {exc}"

    def process_all(self) -> list[str]:
        events: list[str] = []
        while True:
            event = self.process_one()
            if event is None:
                return events
            events.append(event)


def validate_main(argv: list[str] | None = None) -> int:
    """Validate one HDF5 episode, accepting either a file or episode directory."""

    from .recorder import validate_episode_path

    parser = argparse.ArgumentParser(description="Validate one SPD-VR episode")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    report = validate_episode_path(args.path)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


__all__ = [
    "EpisodeCommand",
    "EpisodeCommandType",
    "EpisodeController",
    "EpisodeState",
    "validate_main",
]
