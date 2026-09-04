"""200 Hz dual-arm IK process for the SPD-VR three-window graph.

The process consumes the 1,540-byte PICO tracking stream and publishes the
2×7 arm targets as 272-byte packets.  It never owns the full plant and never
has an API for sending commands to a physical robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import threading
import time
from typing import Any

import numpy as np

from .config import TeleopConfig
from .ik import MuJoCoArmIK
from .robot import RobotSpec
from .teleop import Side, StablePoseCalibration
from .wire import (
    ARM_TARGETS_KEY,
    CONTROL_KEY,
    STATUS_IK_KEY,
    TRACKING_KEY,
    ArmTargetFrame,
    ArmTargetHoldReason,
    ControlCommand,
    ControlSequenceGate,
    TrackingFrame,
    TrackingStreamGate,
    decode_control,
    decode_tracking,
    encode_arm_target,
)


LEFT_VALID = 1
RIGHT_VALID = 2


@dataclass(frozen=True, slots=True)
class ArmIKStatus:
    calibrated: tuple[bool, bool]
    hold_reason: tuple[str, str]
    tracking_epoch: int | None
    sequence: int
    finite: bool
    paused: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "paused" if self.paused else "ready",
            "ready": not self.paused,
            "paused": self.paused,
            "calibrated": list(self.calibrated),
            "hold_reason": list(self.hold_reason),
            "tracking_epoch": self.tracking_epoch,
            "sequence": self.sequence,
            "finite": self.finite,
        }


class DualArmIKController:
    """Keep each arm's alignment, target and HOLD state independent."""

    def __init__(
        self,
        robot: RobotSpec,
        solver: Any,
        *,
        config: TeleopConfig | None = None,
        rate_hz: int = 200,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.robot = robot
        self.solver = solver
        self.config = config or TeleopConfig()
        self.rate_hz = int(rate_hz)
        self._calibration = [
            StablePoseCalibration(self.config),
            StablePoseCalibration(self.config),
        ]
        initial = self.robot.clip(np.zeros(54, dtype=np.float64))
        self._q = {Side.LEFT: initial[:7].copy(), Side.RIGHT: initial[27:34].copy()}
        self._tracking: TrackingFrame | None = None
        self._tracking_gate = TrackingStreamGate()
        self._control_gate = ControlSequenceGate()
        self._epoch: int | None = None
        self._sequence = 0
        self.paused = False
        self._hold_reason = ["disconnected", "disconnected"]

    @property
    def tracking(self) -> TrackingFrame | None:
        return self._tracking

    def reset(self) -> None:
        self._tracking = None
        self._tracking_gate.reset()
        self._epoch = None
        self._sequence = 0
        self.paused = False
        for calibration in self._calibration:
            calibration.reset()
        initial = self.robot.clip(np.zeros(54, dtype=np.float64))
        self._q[Side.LEFT] = initial[:7].copy()
        self._q[Side.RIGHT] = initial[27:34].copy()
        self._hold_reason = ["disconnected", "disconnected"]

    def realign(self) -> None:
        for calibration in self._calibration:
            calibration.reset()

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
            for calibration in self._calibration:
                calibration.reset()
        self._tracking = decoded
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
        command = frame.command
        if command is ControlCommand.PAUSE:
            self.paused = True
        elif command is ControlCommand.RESUME:
            self.paused = False
            self.realign()
        elif command is ControlCommand.REALIGN:
            self.realign()
        elif command is ControlCommand.RESET:
            self.reset()
        elif command is ControlCommand.SHUTDOWN:
            self.paused = True
        return True

    def _side_pose(self, frame: TrackingFrame, index: int) -> tuple[np.ndarray, np.ndarray]:
        hand = frame.left_hand if index == 0 else frame.right_hand
        scale = frame.left_scale if index == 0 else frame.right_scale
        return hand[1, :3].astype(np.float64) * float(scale), hand[1, 3:7].astype(np.float64)

    @staticmethod
    def _slices(index: int) -> tuple[slice, Side]:
        return (slice(0, 7), Side.LEFT) if index == 0 else (slice(27, 34), Side.RIGHT)

    def tick(self, now_ns: int | None = None) -> ArmTargetFrame:
        now = max(1, int(time.monotonic_ns() if now_ns is None else now_ns))
        frame = self._tracking
        source_ns = now if frame is None else max(1, int(frame.source_timestamp_ns))
        epoch = max(1, int(self._epoch or 1))
        age_ns = None if frame is None else now - int(frame.bridge_monotonic_ns)
        stale = age_ns is None or not 0 <= age_ns <= int(self.config.stale_after_ms * 1e6)
        valid_mask = 0
        qdot = {Side.LEFT: np.zeros(7, dtype=np.float64), Side.RIGHT: np.zeros(7, dtype=np.float64)}
        reasons = ["disconnected", "disconnected"]
        if self.paused:
            reasons = ["paused", "paused"]
        for index, side in enumerate((Side.LEFT, Side.RIGHT)):
            if self.paused:
                continue
            if stale:
                reasons[index] = "stale" if frame is not None else "disconnected"
                continue
            active = bool(frame.left_active if index == 0 else frame.right_active)
            if not active:
                reasons[index] = "inactive"
                continue
            calibration = self._calibration[index]
            position, quaternion = self._side_pose(frame, index)
            if not calibration.add(position, quaternion):
                reasons[index] = "aligning"
                continue
            arm_slice, _ = self._slices(index)
            previous = self._q[side]
            try:
                desired = np.asarray(
                    self.solver.solve(
                        side,
                        position - calibration.position,
                        self._relative_quaternion(quaternion, calibration.quaternion),
                        previous,
                    ),
                    dtype=np.float64,
                )
                if desired.shape != (7,) or not np.all(np.isfinite(desired)):
                    raise ValueError("solver returned a non-finite seven-joint vector")
                max_delta = self.robot.velocity[arm_slice] / self.rate_hz
                next_q = previous + np.clip(desired - previous, -max_delta, max_delta)
                # The solver belongs to an arm-only model, so clip with the
                # canonical arm limits before publishing to the full plant.
                next_q = np.clip(next_q, self.robot.lower[arm_slice], self.robot.upper[arm_slice])
                qdot[side] = (next_q - previous) * self.rate_hz
                self._q[side] = next_q
                valid_mask |= LEFT_VALID if index == 0 else RIGHT_VALID
                reasons[index] = "none"
            except Exception:
                reasons[index] = "solver_failure"
        self._hold_reason = reasons
        self._sequence += 1
        return ArmTargetFrame(
            sequence=self._sequence,
            tracking_epoch=epoch,
            source_timestamp_ns=source_ns,
            control_timestamp_ns=now,
            valid_mask=valid_mask,
            left_hold_reason=ArmTargetHoldReason.NONE
            if valid_mask & LEFT_VALID
            else self._reason_enum(reasons[0]),
            right_hold_reason=ArmTargetHoldReason.NONE
            if valid_mask & RIGHT_VALID
            else self._reason_enum(reasons[1]),
            left_q=tuple(float(value) for value in self._q[Side.LEFT]),
            right_q=tuple(float(value) for value in self._q[Side.RIGHT]),
            left_qdot=tuple(float(value) for value in qdot[Side.LEFT]),
            right_qdot=tuple(float(value) for value in qdot[Side.RIGHT]),
        )

    @staticmethod
    def _reason_enum(reason: str) -> ArmTargetHoldReason:
        return {
            "stale": ArmTargetHoldReason.INPUT_STALE,
            "disconnected": ArmTargetHoldReason.DISCONNECTED,
            "inactive": ArmTargetHoldReason.INACTIVE,
            "aligning": ArmTargetHoldReason.ALIGNING,
            "solver_failure": ArmTargetHoldReason.SOLVER_FAILURE,
            "paused": ArmTargetHoldReason.PAUSED,
        }.get(reason, ArmTargetHoldReason.SOLVER_FAILURE)

    @staticmethod
    def _relative_quaternion(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
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

    def status(self) -> ArmIKStatus:
        return ArmIKStatus(
            calibrated=tuple(calibration.position is not None for calibration in self._calibration),
            hold_reason=(self._hold_reason[0], self._hold_reason[1]),
            tracking_epoch=self._epoch,
            sequence=self._sequence,
            finite=bool(np.all(np.isfinite(self._q[Side.LEFT])) and np.all(np.isfinite(self._q[Side.RIGHT]))),
            paused=self.paused,
        )


def run(args: argparse.Namespace) -> int:
    from .zenoh_transport import LatestSample, ZenohNode, peer_config

    urdf = args.urdf.resolve()
    robot = RobotSpec.from_urdf(urdf)
    solver = MuJoCoArmIK(args.model.resolve(), robot)
    controller = DualArmIKController(
        robot,
        solver,
        config=TeleopConfig(urdf_path=str(urdf), alignment_frames=args.alignment_frames),
        rate_hz=args.rate_hz,
    )
    tracking_mailbox = LatestSample()
    control_mailbox = LatestSample()
    stop = threading.Event()
    old_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        with ZenohNode(peer_config(listen=args.listen, endpoint=args.endpoint)) as node:
            node.declare_latest_subscriber(TRACKING_KEY, decode_tracking, tracking_mailbox)
            node.declare_latest_subscriber(CONTROL_KEY, decode_control, control_mailbox)
            publisher = node.declare_publisher(ARM_TARGETS_KEY)
            status_publisher = node.declare_publisher(STATUS_IK_KEY)
            tracking_generation = 0
            control_generation = 0
            tick_period = 1.0 / args.rate_hz
            next_tick = time.monotonic()
            while not stop.is_set():
                sample = tracking_mailbox.take_new(tracking_generation)
                if sample is not None:
                    tracking_generation, value = sample
                    controller.accept_tracking(value)
                control = control_mailbox.take_new(control_generation)
                if control is not None:
                    control_generation, value = control
                    controller.apply_control(value)
                    if value.command is ControlCommand.SHUTDOWN:
                        stop.set()
                target = controller.tick()
                publisher.put(encode_arm_target(target))
                if target.sequence == 1 or target.sequence % max(1, args.status_every) == 0:
                    status_publisher.put(
                        json.dumps(controller.status().as_dict(), sort_keys=True).encode("utf-8")
                    )
                next_tick += tick_period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
                else:
                    next_tick = time.monotonic()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=root / "assets/tianji_wuji2/tianji_wuji2.urdf",
    )
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--rate-hz", type=int, default=200)
    parser.add_argument("--alignment-frames", type=int, default=10)
    parser.add_argument("--status-every", type=int, default=40)
    args = parser.parse_args(argv)
    if args.rate_hz <= 0 or args.alignment_frames <= 0 or args.status_every <= 0:
        parser.error("--rate-hz, --alignment-frames and --status-every must be positive")
    return run(args)


__all__ = ["ArmIKStatus", "DualArmIKController", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
