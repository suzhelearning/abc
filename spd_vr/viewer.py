"""60 Hz MuJoCo viewer/recorder for the SPD-VR three-window graph.

The viewer is the only process that owns the complete plant.  It consumes arm
targets from :mod:`spd_vr.arm_ik`, retargets PICO hands with Wuji2, and writes
the actual MuJoCo state to HDF5.  No physical follower or motor transport is
present in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np

from .config import TeleopConfig
from .data import EpisodeWriter
from .robot import RobotSpec
from .scenes.manifest import load_scene_manifest
from .simulation import SPDVRSim
from .teleop import Side, WujiRetargetAdapter
from .wire import (
    ARM_TARGETS_KEY,
    CONTROL_KEY,
    STATUS_VIEWER_KEY,
    TRACKING_KEY,
    ArmTargetFrame,
    ArmTargetStreamDecoder,
    ControlCommand,
    ControlSequenceGate,
    TrackingFrame,
    TrackingStreamGate,
    decode_arm_target,
    decode_control,
    decode_tracking,
    encode_arm_target,
)


@dataclass(frozen=True, slots=True)
class ViewerStatus:
    tracking_epoch: int | None
    arm_sequence: int | None
    paused: bool
    hand_valid: tuple[bool, bool]
    arm_valid: tuple[bool, bool]
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "paused" if self.paused else "ready",
            "ready": not self.paused,
            "paused": self.paused,
            "tracking_epoch": self.tracking_epoch,
            "arm_sequence": self.arm_sequence,
            "hand_valid": list(self.hand_valid),
            "arm_valid": list(self.arm_valid),
            "finite": self.finite,
        }


class ViewerController:
    """Fuse arm packets and PICO hands without cross-side failure propagation."""

    def __init__(
        self,
        simulation: SPDVRSim,
        robot: RobotSpec,
        hand_retargeter: Any,
        *,
        config: TeleopConfig | None = None,
    ) -> None:
        self.simulation = simulation
        self.robot = robot
        self.hand_retargeter = hand_retargeter
        self.config = config or TeleopConfig()
        self._tracking: TrackingFrame | None = None
        self._tracking_gate = TrackingStreamGate()
        self._arm: ArmTargetFrame | None = None
        self._arm_gate = ArmTargetStreamDecoder(
            max_age_ns=int(self.config.stale_after_ms * 1e6)
        )
        self._control_gate = ControlSequenceGate()
        self._epoch: int | None = None
        self.paused = False
        self.shutdown_requested = False
        self._hand_target = {
            Side.LEFT: self.robot.clip(np.zeros(54, dtype=np.float64))[7:27].copy(),
            Side.RIGHT: self.robot.clip(np.zeros(54, dtype=np.float64))[34:54].copy(),
        }
        # Keep the last accepted arm target independently from the plant's
        # current qpos.  If a stale/invalid packet arrives after physics has
        # moved the arm, HOLD must continue commanding the last safe target
        # rather than silently following the physical rebound.
        initial_qpos = self.simulation.addresses.read_qpos(self.simulation.data)
        self._arm_target = {
            Side.LEFT: initial_qpos[:7].copy(),
            Side.RIGHT: initial_qpos[27:34].copy(),
        }
        self._last_hand_valid = [False, False]
        self._last_arm_valid = [False, False]

    @property
    def tracking(self) -> TrackingFrame | None:
        return self._tracking

    @property
    def arm_target(self) -> ArmTargetFrame | None:
        return self._arm

    def accept_tracking(self, frame: TrackingFrame | bytes | bytearray | memoryview) -> bool:
        try:
            decoded = decode_tracking(frame) if isinstance(frame, (bytes, bytearray, memoryview)) else frame
            if not isinstance(decoded, TrackingFrame):
                raise TypeError("tracking frame must be TrackingFrame or encoded bytes")
            self._tracking_gate.accept(decoded)
        except (TypeError, ValueError, OSError):
            return False
        if self._epoch != decoded.tracking_epoch:
            self._epoch = int(decoded.tracking_epoch)
            # A new bridge epoch invalidates any arm packet from the previous
            # stream.  The arm process will publish fresh targets after its
            # neutral alignment window.
            self._arm = None
            self._arm_gate.reset()
            self._last_arm_valid = [False, False]
            self._reset_hand_filters()
        self._tracking = decoded
        return True

    def accept_arm_target(self, frame: ArmTargetFrame | bytes | bytearray | memoryview) -> bool:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            try:
                decoded = self._arm_gate.decode(bytes(frame))
            except ValueError:
                return False
        else:
            if not isinstance(frame, ArmTargetFrame):
                # Zenoh callbacks should already decode to ArmTargetFrame, but
                # callers may hand malformed objects to this boundary during
                # replay/tests.  Treat them exactly like a corrupt packet and
                # retain the last safe command instead of escaping the loop.
                return False
            try:
                # Run the same sequence/epoch gate as the byte path.  Zenoh
                # callbacks decode once before entering this controller.
                decoded = self._arm_gate.decode(encode_arm_target(frame))
            except ValueError:
                return False
        if self._epoch is not None and decoded.tracking_epoch != self._epoch:
            # Keep the last safe target until the arm stream catches up to the
            # currently active PICO epoch.
            return False
        self._arm = decoded
        return True

    def apply_control(self, frame: Any) -> bool:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            try:
                frame = decode_control(frame)
            except ValueError:
                return False
        try:
            accepted = self._control_gate.accept(frame)
        except ValueError:
            return False
        if not accepted:
            return False
        if frame.command is ControlCommand.PAUSE:
            self.paused = True
        elif frame.command is ControlCommand.RESUME:
            self.paused = False
        elif frame.command is ControlCommand.RESET:
            self.paused = False
            self._tracking = None
            self._tracking_gate.reset()
            self._arm = None
            self._arm_gate.reset()
            self._epoch = None
            self._last_hand_valid = [False, False]
            self._last_arm_valid = [False, False]
            self.simulation.reset()
            current = self.simulation.addresses.read_qpos(self.simulation.data)
            self._arm_target[Side.LEFT] = current[:7].copy()
            self._arm_target[Side.RIGHT] = current[27:34].copy()
            self._hand_target[Side.LEFT] = current[7:27].copy()
            self._hand_target[Side.RIGHT] = current[34:54].copy()
            self._reset_hand_filters()
        elif frame.command is ControlCommand.SHUTDOWN:
            self.shutdown_requested = True
        return True

    def _reset_hand_filters(self) -> None:
        reset = getattr(self.hand_retargeter, "reset", None)
        if reset is not None:
            for side in (Side.LEFT, Side.RIGHT):
                try:
                    reset(side)
                except Exception:
                    # A filter reset is best effort; the side remains held
                    # until a finite retarget result is accepted.
                    pass

    def _tracking_fresh(self, now_ns: int, frame: TrackingFrame | None) -> bool:
        if frame is None:
            return False
        age_ns = now_ns - int(frame.bridge_monotonic_ns)
        return 0 <= age_ns <= int(self.config.stale_after_ms * 1e6)

    def _arm_fresh(self, now_ns: int, frame: ArmTargetFrame | None) -> bool:
        if frame is None:
            return False
        age_ns = now_ns - int(frame.control_timestamp_ns)
        return 0 <= age_ns <= int(self.config.stale_after_ms * 1e6)

    def step(self, now_ns: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply one 60 Hz target and advance the plant.

        Returns ``(validity, pico_hands, hand_scale)`` for the recorder.  A
        paused controller does not step physics, preserving the exact pause
        contract used by data collection.
        """
        now = max(1, int(time.monotonic_ns() if now_ns is None else now_ns))
        if self.paused:
            frame = self._tracking
            hands = (
                np.stack((frame.left_hand, frame.right_hand))
                if frame is not None
                else np.zeros((2, 26, 7), dtype=np.float32)
            )
            scale = (
                np.asarray((frame.left_scale, frame.right_scale), dtype=np.float32)
                if frame is not None
                else np.ones(2, dtype=np.float32)
            )
            return np.zeros(2, dtype=np.bool_), hands, scale

        target = self.simulation.addresses.read_qpos(self.simulation.data)
        tracking_fresh = self._tracking_fresh(now, self._tracking)
        arm_fresh = self._arm_fresh(now, self._arm)
        hand_valid = [False, False]
        arm_valid = [False, False]
        for index, side in enumerate((Side.LEFT, Side.RIGHT)):
            arm_slice = slice(0, 7) if index == 0 else slice(27, 34)
            hand_slice = slice(7, 27) if index == 0 else slice(34, 54)
            # Start from the side's last safe arm command.  A fresh packet
            # may replace it below; stale, invalid, or missing packets leave
            # this value untouched.
            target[arm_slice] = self._arm_target[side]
            if arm_fresh and self._arm is not None and self._arm.valid_mask & (1 << index):
                values = np.asarray(
                    self._arm.left_q if index == 0 else self._arm.right_q,
                    dtype=np.float64,
                )
                if values.shape == (7,) and np.all(np.isfinite(values)):
                    self._arm_target[side] = np.clip(
                        values, self.robot.lower[arm_slice], self.robot.upper[arm_slice]
                    )
                    target[arm_slice] = self._arm_target[side]
                    arm_valid[index] = True
            if tracking_fresh and self._tracking is not None:
                active = bool(
                    self._tracking.left_active if index == 0 else self._tracking.right_active
                )
                if active:
                    hand = self._tracking.left_hand if index == 0 else self._tracking.right_hand
                    scale = self._tracking.left_scale if index == 0 else self._tracking.right_scale
                    try:
                        points = np.asarray(hand, dtype=np.float64).copy()
                        points[:, :3] *= float(scale)
                        desired = np.asarray(self.hand_retargeter.retarget(side, points), dtype=np.float64)
                        if desired.shape != (20,) or not np.all(np.isfinite(desired)):
                            raise ValueError("Wuji retargeter returned an invalid 20-D hand target")
                        previous = self._hand_target[side]
                        max_delta = self.robot.velocity[hand_slice] / self.config.control_hz
                        next_hand = previous + np.clip(desired - previous, -max_delta, max_delta)
                        self._hand_target[side] = np.clip(
                            next_hand, self.robot.lower[hand_slice], self.robot.upper[hand_slice]
                        )
                        target[hand_slice] = self._hand_target[side]
                        hand_valid[index] = True
                    except Exception:
                        target[hand_slice] = self._hand_target[side]
            else:
                target[hand_slice] = self._hand_target[side]
        self._last_hand_valid = hand_valid
        self._last_arm_valid = arm_valid
        self.simulation.set_target(target)
        self.simulation.step()
        if self._tracking is None:
            hands = np.zeros((2, 26, 7), dtype=np.float32)
            hands[..., 6] = 1.0
            scale = np.ones(2, dtype=np.float32)
        else:
            hands = np.stack((self._tracking.left_hand, self._tracking.right_hand)).astype(np.float32)
            scale = np.asarray((self._tracking.left_scale, self._tracking.right_scale), dtype=np.float32)
        # A training row is valid only when the arm and hand halves of that
        # side are both fresh.  The actual target is still held independently.
        validity = np.asarray(
            (hand_valid[0] and arm_valid[0], hand_valid[1] and arm_valid[1]),
            dtype=np.bool_,
        )
        return validity, hands, scale

    def status(self) -> ViewerStatus:
        target = self.simulation.addresses.read_qpos(self.simulation.data)
        return ViewerStatus(
            tracking_epoch=self._epoch,
            arm_sequence=None if self._arm is None else self._arm.sequence,
            paused=self.paused,
            hand_valid=(self._last_hand_valid[0], self._last_hand_valid[1]),
            arm_valid=(self._last_arm_valid[0], self._last_arm_valid[1]),
            finite=bool(np.all(np.isfinite(target))),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    paths = [
        *root.glob("spd_vr/**/*.py"),
        *root.glob("wuji_retargeting/**/*.py"),
        root / "abc_minimal/dit.py",
        root / "abc_minimal/flow.py",
    ]
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def _preflight_recording(model_path: Path, urdf_path: Path, scene_manifest: dict[str, Any] | None) -> None:
    from .model_compiler.artifacts import verify_artifacts, verify_contact_qualified

    verify_contact_qualified(model_path.parent / "collision_manifest.yaml", urdf_path=urdf_path)
    verified = verify_artifacts(model_path.parent / "model_manifest.yaml", urdf_path)
    if scene_manifest is None and verified.full_model.resolve() != model_path.resolve():
        raise ValueError("recording without --scene-manifest requires verified unified_plant.xml")


def run(args: argparse.Namespace) -> int:
    from .zenoh_transport import LatestSample, ZenohNode, peer_config

    model_path = args.model.resolve()
    urdf_path = args.urdf.resolve()
    root = Path(__file__).resolve().parents[1]
    scene_manifest = load_scene_manifest(args.scene_manifest) if args.scene_manifest else None
    if scene_manifest is not None:
        scene_model = scene_manifest.get("model")
        if (
            isinstance(scene_model, dict)
            and scene_model.get("sha256") is not None
            and scene_model.get("sha256") != _sha256(model_path)
        ):
            raise ValueError("scene manifest model hash does not match --model")
        if not args.object_body:
            names = scene_manifest.get("object_bodies", [])
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("scene manifest object_bodies must be a list of strings")
            args.object_body = list(names)
        source_hashes = scene_manifest.get("builder_source_sha256", {})
        required_sources = {"registry.py", "scene_builder.py", "model_scene.py"}
        if (
            not isinstance(source_hashes, dict)
            or not required_sources.issubset(source_hashes)
        ):
            raise ValueError("scene builder source hashes are incomplete")
        for name, expected in source_hashes.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"scene builder source path is malformed: {name!r}")
            if not isinstance(expected, str) or len(expected) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in expected
            ):
                raise ValueError(f"scene builder source hash is malformed: {name}")
            source = root / "spd_vr/scenes" / name
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError(f"scene builder source hash mismatch: {name}")
    if args.record_to is not None:
        if args.backend != "mujoco":
            raise ValueError("recording requires --backend mujoco")
        _preflight_recording(model_path, urdf_path, scene_manifest)
    robot = RobotSpec.from_urdf(urdf_path)
    retargeter = WujiRetargetAdapter(str(args.left_hand_config), str(args.right_hand_config))
    config = TeleopConfig(urdf_path=str(urdf_path))
    writer = None
    if args.record_to is not None:
        writer = EpisodeWriter(
            args.record_to,
            {
                "system": "PICO 4 Ultra + Tianji dual arms + Wuji2 dual hands",
                "output_boundary": "MuJoCo simulation only",
                "model_sha256": _sha256(model_path),
                "urdf_sha256": _sha256(urdf_path),
                "retarget_config_sha256": {
                    "left": _sha256(args.left_hand_config.resolve()),
                    "right": _sha256(args.right_hand_config.resolve()),
                },
                "source_sha256": _source_hashes(root),
                "runtime": {
                    "backend": args.backend,
                    "endpoint": args.endpoint,
                    "object_bodies": list(args.object_body),
                    "process_graph": "pxrea_bridge -> arm_ik + viewer",
                },
                "canonical_joint_order": list(robot.joint_names),
                "scene_manifest": scene_manifest,
            },
            overwrite=args.overwrite,
        )
    stop = threading.Event()
    old_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    failed = False
    viewer = None
    try:
        with ZenohNode(peer_config(listen=args.listen, endpoint=args.endpoint)) as node, SPDVRSim(
            model_path,
            robot,
            backend=args.backend,
            gpu_id=args.gpu_id,
            object_bodies=args.object_body,
        ) as simulation:
            controller = ViewerController(simulation, robot, retargeter, config=config)
            tracking_mailbox = LatestSample()
            arm_mailbox = LatestSample()
            control_mailbox = LatestSample()
            node.declare_latest_subscriber(TRACKING_KEY, decode_tracking, tracking_mailbox)
            node.declare_latest_subscriber(ARM_TARGETS_KEY, decode_arm_target, arm_mailbox)
            node.declare_latest_subscriber(CONTROL_KEY, decode_control, control_mailbox)
            status_publisher = node.declare_publisher(STATUS_VIEWER_KEY)
            if scene_manifest is not None:
                simulation.reset_scene(scene_manifest["reset"])
            if args.viewer:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)
            tracking_generation = arm_generation = control_generation = 0
            next_tick = time.monotonic()
            while not stop.is_set() and (viewer is None or viewer.is_running()):
                sample = tracking_mailbox.take_new(tracking_generation)
                if sample is not None:
                    tracking_generation, value = sample
                    controller.accept_tracking(value)
                sample = arm_mailbox.take_new(arm_generation)
                if sample is not None:
                    arm_generation, value = sample
                    controller.accept_arm_target(value)
                sample = control_mailbox.take_new(control_generation)
                if sample is not None:
                    control_generation, value = sample
                    controller.apply_control(value)
                if controller.shutdown_requested:
                    stop.set()
                validity, hands, scale = controller.step()
                if viewer is not None:
                    viewer.sync()
                if writer is not None and controller.tracking is not None and not controller.paused:
                    frame = controller.tracking
                    writer.append(
                        simulation.record_frame(
                            time.monotonic_ns(),
                            hands,
                            frame.source_timestamp_ns,
                            frame.bridge_monotonic_ns,
                            frame.sequence,
                            frame.tracking_epoch,
                            scale,
                            validity,
                        )
                    )
                status_publisher.put(
                    json.dumps(controller.status().as_dict(), sort_keys=True).encode("utf-8")
                )
                next_tick += 1.0 / config.control_hz
                delay = next_tick - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
                else:
                    next_tick = time.monotonic()
    except BaseException:
        failed = True
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            if failed or not writer.frame_count:
                writer.abort()
            else:
                print(f"published {writer.finish()}")
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=root / "assets/tianji_wuji2/tianji_wuji2.urdf")
    parser.add_argument("--left-hand-config", type=Path, default=root / "spd_vr/config/wuji2_pico_left.yaml")
    parser.add_argument("--right-hand-config", type=Path, default=root / "spd_vr/config/wuji2_pico_right.yaml")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    listen = parser.add_mutually_exclusive_group()
    listen.add_argument("--listen", dest="listen", action="store_true", default=True)
    listen.add_argument("--no-listen", dest="listen", action="store_false")
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--object-body", action="append", default=[])
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--record-to", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    return run(parser.parse_args(argv))


__all__ = ["ViewerController", "ViewerStatus", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
