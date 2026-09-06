import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spd_vr.arm_ik import DualArmIKController
from spd_vr.alignment import SideAlignment
from spd_vr.config import TeleopConfig
from spd_vr.episode import EpisodeCommandType, EpisodeController, EpisodeState
from spd_vr.robot import RobotSpec
from spd_vr.teleop import Side
from spd_vr.viewer import ViewerController
from spd_vr.wire import (
    ArmTargetHoldReason,
    ControlCommand,
    ControlFrame,
    TrackingFrame,
    encode_control,
)


def _tracking(sequence: int, *, left_active: bool = True, right_active: bool = True, epoch: int = 1):
    pose = np.zeros((26, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    return TrackingFrame(
        sequence=sequence,
        tracking_epoch=epoch,
        source_timestamp_ns=100 + sequence,
        bridge_monotonic_ns=100 + sequence,
        left_active=left_active,
        right_active=right_active,
        head_valid=False,
        left_scale=1.0,
        right_scale=1.0,
        head_pose=np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.float32),
        left_hand=pose,
        right_hand=pose,
    )


class _FakeArmSolver:
    def solve(self, side, wrist_position, wrist_quaternion_xyzw, previous_qpos):
        return np.asarray(previous_qpos, dtype=np.float64) + 0.2


def test_dual_arm_controller_calibrates_and_holds_each_side_independently(vendor_urdf):
    robot = RobotSpec.from_urdf(vendor_urdf)
    controller = DualArmIKController(
        robot,
        _FakeArmSolver(),
        config=TeleopConfig(alignment_frames=2),
        rate_hz=200,
    )
    assert controller.accept_tracking(_tracking(1))
    first = controller.tick(now_ns=101)
    assert first.valid_mask == 0
    assert first.left_hold_reason is ArmTargetHoldReason.ALIGNING
    assert controller.accept_tracking(_tracking(2))
    second = controller.tick(now_ns=102)
    assert second.valid_mask == 3
    assert np.max(np.abs(second.left_qdot)) > 0
    assert controller.accept_tracking(_tracking(3, left_active=False))
    third = controller.tick(now_ns=103)
    assert third.valid_mask == 2
    assert third.left_hold_reason is ArmTargetHoldReason.INACTIVE
    np.testing.assert_allclose(third.left_q, second.left_q)
    assert third.right_q != second.right_q
    assert controller.apply_control(
        encode_control(ControlFrame(1, 1_000, ControlCommand.PAUSE))
    )
    paused = controller.tick(now_ns=104)
    assert paused.left_hold_reason is ArmTargetHoldReason.PAUSED


class _Addresses:
    def __init__(self):
        self.value = np.zeros(54, dtype=np.float64)

    def read_qpos(self, data):
        return self.value.copy()


class _Simulation:
    def __init__(self):
        self.addresses = _Addresses()
        self.data = object()
        self.targets = []
        self.steps = 0

    def set_target(self, target):
        self.addresses.value = np.asarray(target, dtype=np.float64).copy()
        self.targets.append(self.addresses.value.copy())

    def step(self):
        self.steps += 1

    def reset(self):
        self.addresses.value.fill(0)


class _FakeHands:
    def retarget(self, side, points):
        return np.full(20, 0.1 if side is Side.LEFT else -0.1)


def test_viewer_fuses_arm_packet_and_hands_then_holds_on_stale_input(vendor_urdf):
    robot = RobotSpec.from_urdf(vendor_urdf)
    simulation = _Simulation()
    viewer = ViewerController(
        simulation,
        robot,
        _FakeHands(),
        config=TeleopConfig(stale_after_ms=50),
    )
    tracking = _tracking(1)
    arm_controller = DualArmIKController(
        robot,
        _FakeArmSolver(),
        config=TeleopConfig(alignment_frames=1),
    )
    arm_controller.accept_tracking(tracking)
    arm_target = arm_controller.tick(now_ns=101)
    assert viewer.accept_tracking(tracking)
    assert viewer.accept_arm_target(arm_target)
    validity, hands, scale = viewer.step(now_ns=101)
    assert validity.tolist() == [True, True]
    assert hands.shape == (2, 26, 7)
    assert scale.tolist() == [1.0, 1.0]
    old_target = simulation.targets[-1].copy()
    validity, _, _ = viewer.step(now_ns=100_000_102)
    assert validity.tolist() == [False, False]
    np.testing.assert_allclose(simulation.targets[-1], old_target)
    assert simulation.steps == 2


def test_viewer_arm_hold_uses_last_command_after_physics_moves(vendor_urdf):
    class MovingSimulation(_Simulation):
        def step(self):
            super().step()
            # Simulate passive dynamics moving the plant away from the last
            # command between control ticks.  HOLD must not follow this drift.
            self.addresses.value[0] += 0.25
            self.addresses.value[27] -= 0.25

    robot = RobotSpec.from_urdf(vendor_urdf)
    simulation = MovingSimulation()
    viewer = ViewerController(
        simulation,
        robot,
        _FakeHands(),
        config=TeleopConfig(stale_after_ms=50),
    )
    tracking = _tracking(1)
    arm_controller = DualArmIKController(
        robot,
        _FakeArmSolver(),
        config=TeleopConfig(alignment_frames=1),
    )
    arm_controller.accept_tracking(tracking)
    arm_target = arm_controller.tick(now_ns=101)
    viewer.accept_tracking(tracking)
    viewer.accept_arm_target(arm_target)
    viewer.step(now_ns=101)
    command = simulation.targets[-1].copy()

    validity, _, _ = viewer.step(now_ns=100_000_102)
    assert validity.tolist() == [False, False]
    np.testing.assert_allclose(simulation.targets[-1], command)


def test_viewer_holds_future_timestamp_packets(vendor_urdf):
    robot = RobotSpec.from_urdf(vendor_urdf)
    simulation = _Simulation()
    viewer = ViewerController(simulation, robot, _FakeHands(), config=TeleopConfig(stale_after_ms=50))
    tracking = _tracking(1)
    arm_controller = DualArmIKController(robot, _FakeArmSolver(), config=TeleopConfig(alignment_frames=1))
    arm_controller.accept_tracking(tracking)
    arm_target = arm_controller.tick(now_ns=101)
    assert viewer.accept_tracking(tracking)
    assert viewer.accept_arm_target(arm_target)
    validity, _, _ = viewer.step(now_ns=100)
    assert validity.tolist() == [False, False]


def test_side_alignment_holds_malformed_epoch_timestamp_or_active_flag():
    alignment = SideAlignment(stable_frames=1)
    pose = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    malformed = alignment.accept(pose, "true", 1, 1)
    assert malformed.hold_reason == "invalid_metadata"
    malformed = alignment.accept(pose, True, "1", 2)
    assert malformed.hold_reason == "invalid_metadata"
    malformed = alignment.accept(pose, True, 1, "2")
    assert malformed.hold_reason == "invalid_metadata"
    accepted = alignment.accept(pose, True, 1, 2)
    assert accepted.aligned and accepted.hold_reason is None
    assert alignment.stale("2").hold_reason == "invalid_metadata"


def test_side_alignment_stale_gate_holds_only_after_freshness_budget():
    alignment = SideAlignment(stable_frames=1, stale_after_ns=50)
    pose = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert alignment.stale(0).hold_reason == "stale"
    accepted = alignment.accept(pose, True, 1, 100)
    assert accepted.aligned and accepted.hold_reason is None
    assert alignment.stale(150).hold_reason is None
    assert alignment.stale(151).hold_reason == "stale"
    assert alignment.stale(99).hold_reason == "stale"


def test_invalid_tracking_bytes_are_dropped_without_resetting_safe_state(vendor_urdf):
    robot = RobotSpec.from_urdf(vendor_urdf)
    arm = DualArmIKController(robot, _FakeArmSolver())
    simulation = _Simulation()
    viewer = ViewerController(simulation, robot, _FakeHands())
    assert arm.accept_tracking(b"invalid") is False
    assert viewer.accept_tracking(b"invalid") is False
    assert arm.status().hold_reason == ("disconnected", "disconnected")
    assert viewer.status().finite is True


class _EpisodeSimulator:
    def __init__(self):
        self.paused = False
        self.contact = True
        self._target = np.zeros(1)
        self.addresses = SimpleNamespace(actuator=np.asarray([0], dtype=np.int64))
        self.data = SimpleNamespace(
            qpos=np.zeros(1), qvel=np.zeros(1), act=None, ctrl=np.zeros(1),
            mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)), time=0.0,
        )
        self.set_paused_calls = []

    def set_paused(self, value):
        self.paused = bool(value)
        self.set_paused_calls.append(self.paused)

    def has_task_object_contact(self):
        return self.contact


class _EpisodeTask:
    def reset(self, seed):
        return SimpleNamespace(manifest=lambda: {"scene": "test", "seed": seed}, objects=[])


class _EpisodeRecorder:
    def __init__(self):
        self.manifest = None
        self.finished = 0
        self.discarded = []

    def start_episode(self, episode_id, manifest):
        self.manifest = manifest

    def finish_episode(self):
        self.finished += 1

    def discard_episode(self, reason):
        self.discarded.append(reason)


def test_episode_state_machine_guards_contact_and_supports_revert_pause_skip():
    simulation = _EpisodeSimulator()
    recorder = _EpisodeRecorder()
    controller = EpisodeController(
        simulation,
        _EpisodeTask(),
        recorder=recorder,
        collection_identity={
            "run_id": "run-001",
            "operator_id": "operator-01",
            "pico_serial": "pico-01",
        },
    )
    controller.enqueue(EpisodeCommandType.START)
    assert controller.process_one() == "started"
    assert recorder.manifest["collection"] == {
        "run_id": "run-001",
        "operator_id": "operator-01",
        "pico_serial": "pico-01",
    }
    controller.enqueue(EpisodeCommandType.CHECKPOINT)
    assert controller.process_one() == "checkpoint rejected while hand-object contact is active"
    simulation.contact = False
    controller.enqueue(EpisodeCommandType.CHECKPOINT)
    assert controller.process_one() == "checkpointed"
    simulation.data.qpos[0] = 4.0
    simulation._target[0] = 7.0
    controller.enqueue(EpisodeCommandType.REVERT)
    assert controller.process_one() == "reverted"
    assert simulation.data.qpos[0] == 0.0
    assert simulation._target[0] == 0.0
    controller.enqueue(EpisodeCommandType.PAUSE)
    assert controller.process_one() == "paused"
    assert simulation.paused is True
    controller.enqueue(EpisodeCommandType.SKIP)
    assert controller.process_one() == "skipped"
    assert controller.state is EpisodeState.IDLE
    assert simulation.paused is False
    assert recorder.discarded == ["operator_skip"]


def test_foreground_launcher_fake_source_shutdown_option_is_wired():
    """Run the complete graph in one foreground supervisor without tmux."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "start_spd_vr.sh"
    help_result = subprocess.run(
        [str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "foreground supervisor" in help_result.stdout
    assert "tmux" not in help_result.stdout.lower()

    result = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--viewer",
            "--fake-source-jsonl",
            "/tmp/spd-events.jsonl",
            "--wait-for-shutdown",
            "--require-usable-training",
            "--record-to",
            "/tmp/spd-episode.hdf5",
            "--collection-run-id",
            "run-001",
            "--operator-id",
            "operator-01",
            "--pico-serial",
            "pico-01",
            "--collection-plan",
            "/tmp/collection-plan.json",
            "--episode-id",
            "jenga--hollow_tower-0001",
            "--scene-manifest",
            "/tmp/scene.json",
            "--model",
            "/tmp/unified.xml",
            "--arm-model",
            "/tmp/arm.xml",
            "--urdf",
            "/tmp/robot.urdf",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "mode=foreground" in result.stdout
    assert "pxrea_bridge:" in result.stdout
    viewer_line = next(
        line for line in result.stdout.splitlines() if line.startswith("viewer:")
    )
    assert "--viewer" in viewer_line
    assert "--fake-source-jsonl /tmp/spd-events.jsonl" in result.stdout
    assert "--wait-for-shutdown" in result.stdout
    assert "--require-usable-training" in result.stdout
    assert "--collection-run-id run-001" in result.stdout
    assert "--operator-id operator-01" in result.stdout
    assert "--pico-serial pico-01" in result.stdout
    assert "--collection-plan /tmp/collection-plan.json" in result.stdout
    assert "--episode-id jenga--hollow_tower-0001" in result.stdout

    rejected = subprocess.run(
        [str(script), "--dry-run", "--wait-for-shutdown"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "only valid with --fake-source-jsonl" in rejected.stderr

    invalid_timeout = subprocess.run(
        [str(script), "--dry-run", "--router-timeout", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_timeout.returncode == 2
    assert "--router-timeout must be a positive integer" in invalid_timeout.stderr

    missing_record = subprocess.run(
        [str(script), "--dry-run", "--require-usable-training"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_record.returncode == 2
    assert "requires --record-to" in missing_record.stderr
    partial_collection = subprocess.run(
        [str(script), "--dry-run", "--collection-run-id", "run-001"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert partial_collection.returncode == 2
    assert "must be supplied together" in partial_collection.stderr

    collection_without_record = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--collection-run-id",
            "run-001",
            "--operator-id",
            "operator-01",
            "--pico-serial",
            "pico-01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert collection_without_record.returncode == 2
    assert "requires --record-to" in collection_without_record.stderr

    partial_plan = subprocess.run(
        [str(script), "--dry-run", "--collection-plan", "/tmp/plan.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert partial_plan.returncode == 2
    assert "must be supplied together" in partial_plan.stderr

    plan_without_scene = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--collection-plan",
            "/tmp/plan.json",
            "--episode-id",
            "jenga--hollow_tower-0001",
            "--record-to",
            "/tmp/episode.hdf5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert plan_without_scene.returncode == 2
    assert "requires --record-to and --scene-manifest" in plan_without_scene.stderr


def test_collection_launcher_derives_formal_run_arguments(tmp_path):
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    (run_dir / "collection-plan.json").write_text(
        json.dumps(
            {
                "collection_identity": {
                    "run_id": "run-001",
                    "operator_id": "operator-01",
                    "pico_serial": "pico-01",
                },
                "episodes": [
                    {
                        "episode_id": "jenga--hollow_tower-0001",
                        "task": "jenga/hollow_tower",
                        "seed": 1000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_spd_vr.sh"
    result = subprocess.run(
        [
            str(script),
            str(run_dir),
            "jenga--hollow_tower-0001",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "mode=foreground" in result.stdout
    assert f"--collection-plan {run_dir}/collection-plan.json" in result.stdout
    assert "--episode-id jenga--hollow_tower-0001" in result.stdout
    assert "--collection-run-id run-001" in result.stdout
    assert "--operator-id operator-01" in result.stdout
    assert "--pico-serial pico-01" in result.stdout
    assert f"--record-to {run_dir}/jenga--hollow_tower-0001.hdf5" in result.stdout
