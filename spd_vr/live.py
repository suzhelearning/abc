"""Run the simulation-only PICO → IK/retarget → 54-DoF MuJoCo loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .config import TeleopConfig
from .collection import collection_metadata
from .collection_plan import load_collection_plan, planned_episode
from .data import EpisodeWriter
from .ik import MuJoCoArmIK
from .robot import RobotSpec
from .simulation import SPDVRSim
from .scenes.manifest import load_scene_manifest
from .teleop import PicoFrame, TeleopMapper, WujiRetargetAdapter
from .wire import TRACKING_KEY, TrackingStreamGate, decode_tracking


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _pico_frame(frame: object) -> PicoFrame:
    hands = np.stack((frame.left_hand, frame.right_hand)).astype(np.float64)
    scale = np.asarray((frame.left_scale, frame.right_scale), dtype=np.float64)
    wrist_position = hands[:, 1, :3] * scale[:, None]
    return PicoFrame(
        timestamp_ns=int(frame.bridge_monotonic_ns),
        source_timestamp_ns=int(frame.source_timestamp_ns),
        sequence_id=int(frame.sequence),
        tracking_epoch=int(frame.tracking_epoch),
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=hands[:, 1, 3:7],
        hands=hands,
        hand_scale=scale,
        valid=np.asarray((frame.left_active, frame.right_active)),
    )


def run(args: argparse.Namespace) -> int:
    from .zenoh_transport import LatestSample, ZenohNode, peer_config

    if args.record_to is not None and args.backend != "mujoco":
        raise ValueError(
            "recording requires --backend mujoco until MJWarp contact-force state is synchronized"
        )
    model_path = args.model.resolve()
    arm_model_path = args.arm_model.resolve()
    urdf_path = args.urdf.resolve()
    left_hand_config = args.left_hand_config.resolve()
    right_hand_config = args.right_hand_config.resolve()
    root = Path(__file__).resolve().parents[1]
    scene_manifest: dict[str, object] | None = None
    if args.scene_manifest is not None:
        scene_manifest = load_scene_manifest(args.scene_manifest)
        scene_model = scene_manifest.get("model")
        if isinstance(scene_model, dict):
            scene_hash = scene_model.get("sha256")
            if scene_hash is not None and (not isinstance(scene_hash, str) or scene_hash != _sha256(model_path)):
                raise ValueError("scene manifest model hash does not match --model")
        if not args.object_body:
            object_bodies = scene_manifest.get("object_bodies", [])
            if not isinstance(object_bodies, list) or not all(isinstance(name, str) for name in object_bodies):
                raise ValueError("scene manifest object_bodies must be a list of strings")
            args.object_body = list(object_bodies)
        source_hashes = scene_manifest.get("builder_source_sha256")
        if isinstance(source_hashes, dict):
            scene_root = root / "spd_vr" / "scenes"
            for name, expected_hash in source_hashes.items():
                if not isinstance(name, str) or Path(name).name != name or not isinstance(expected_hash, str):
                    raise ValueError("scene builder source hashes are malformed")
                source_path = scene_root / name
                if not source_path.is_file() or _sha256(source_path) != expected_hash:
                    raise ValueError(f"scene builder source hash mismatch: {name}")
    collection_plan = None
    planned = None
    if args.collection_plan is not None:
        if args.record_to is None:
            raise ValueError("--collection-plan requires --record-to")
        if not isinstance(args.episode_id, str) or not args.episode_id.strip():
            raise ValueError("--collection-plan requires --episode-id")
        if scene_manifest is None:
            raise ValueError("--collection-plan requires --scene-manifest")
        collection_plan = load_collection_plan(args.collection_plan)
        planned = planned_episode(collection_plan, args.episode_id.strip())
        if scene_manifest.get("task") != planned["task"]:
            raise ValueError(
                "scene manifest task does not match collection-plan episode "
                f"{planned['episode_id']}: {planned['task']}"
            )
        reset = scene_manifest.get("reset")
        if not isinstance(reset, dict) or reset.get("seed") != planned["seed"]:
            raise ValueError("scene manifest seed does not match collection-plan episode")
    elif args.episode_id is not None:
        raise ValueError("--episode-id requires --collection-plan")
    if args.record_to is not None:
        from .model_compiler.qualification import verify_contact_qualification_receipt

        verify_contact_qualification_receipt(model_path.parent, urdf_path)
        # Verify the authoritative base artifacts as well.  A scene XML is
        # allowed to add free task bodies, but it must live beside a verified
        # compiler output and carry its own scene-manifest model hash.
        if scene_manifest is None and (model_path.parent / "unified_plant.xml").resolve() != model_path:
            raise ValueError("recording without --scene-manifest requires the verified unified_plant.xml")
    teleop_config = TeleopConfig(urdf_path=str(urdf_path))
    robot = RobotSpec.from_urdf(urdf_path)
    arm_ik = MuJoCoArmIK(arm_model_path, robot)
    retarget = WujiRetargetAdapter(str(left_hand_config), str(right_hand_config))
    mapper = TeleopMapper(robot, arm_ik, retarget, teleop_config)
    mailbox = LatestSample()
    stream_gate = TrackingStreamGate()
    identity = collection_metadata(
        run_id=args.collection_run_id,
        operator_id=args.operator_id,
        pico_serial=args.pico_serial,
    )
    if planned is not None:
        planned_identity = collection_plan.get("collection_identity")
        if planned_identity is not None and identity != planned_identity:
            raise ValueError("runtime collection identity does not match collection plan")
    if identity and args.record_to is None:
        raise ValueError("collection metadata requires --record-to")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    writer = None
    if args.record_to is not None:
        manifest = {
            "system": "PICO 4 Ultra + Tianji dual arms + Wuji2 dual hands",
            "output_boundary": "MuJoCo simulation only",
            "model_sha256": _sha256(model_path),
            "arm_model_sha256": _sha256(arm_model_path),
            "urdf_sha256": _sha256(urdf_path),
            "collision_manifest_sha256": _sha256(model_path.parent / "collision_manifest.yaml"),
            "retarget_config_sha256": {
                "left": _sha256(left_hand_config),
                "right": _sha256(right_hand_config),
            },
            "source_sha256": _source_hashes(root),
            "teleop_config": asdict(teleop_config),
            "runtime": {
                "backend": args.backend,
                "endpoint": args.endpoint,
                "object_bodies": list(args.object_body),
            },
            "canonical_joint_order": list(robot.joint_names),
            "scene_manifest": scene_manifest,
        }
        if planned is not None:
            plan_path = args.collection_plan.resolve()
            manifest["episode_id"] = planned["episode_id"]
            manifest["collection_plan"] = {
                "path": str(plan_path),
                "sha256": _sha256(plan_path),
                "episode_id": planned["episode_id"],
                "seed": planned["seed"],
                "task": planned["task"],
            }
        if identity:
            manifest["collection"] = identity
        writer = EpisodeWriter(
            args.record_to,
            manifest,
            overwrite=args.overwrite,
            require_usable_training=args.require_usable_training,
        )

    viewer = None
    generation = 0
    latest = None
    latest_scale = np.ones(2, dtype=np.float32)
    latest_command_valid = np.zeros(2, dtype=np.bool_)
    next_tick = time.monotonic()
    failed = False
    try:
        with ZenohNode(peer_config(listen=True, endpoint=args.endpoint)) as node, SPDVRSim(
            model_path,
            robot,
            backend=args.backend,
            gpu_id=args.gpu_id,
            object_bodies=args.object_body,
        ) as simulation:
            node.declare_latest_subscriber(TRACKING_KEY, decode_tracking, mailbox)
            simulation.reset()
            if scene_manifest is not None:
                simulation.reset_scene(scene_manifest["reset"])
            if args.viewer:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)
            while not stop.is_set() and (viewer is None or viewer.is_running()):
                sample = mailbox.take_new(generation)
                if sample is not None:
                    generation, tracking = sample
                    try:
                        stream_gate.accept(tracking)
                        latest = _pico_frame(tracking)
                        latest_scale = latest.hand_scale.astype(np.float32)
                        target, _status = mapper.update(latest, now_ns=time.monotonic_ns())
                        latest_command_valid = np.asarray(
                            (target.left_valid, target.right_valid), dtype=np.bool_
                        )
                        simulation.set_target(target)
                    except ValueError:
                        # Malformed/out-of-order input cannot perturb the last safe target.
                        pass
                simulation.step()
                if viewer is not None:
                    if simulation.backend == "mjwarp":
                        simulation.sync_for_viewer()
                    viewer.sync()
                if writer is not None and latest is not None:
                    age_ns = time.monotonic_ns() - latest.timestamp_ns
                    validity = latest_command_valid & (
                        age_ns <= int(teleop_config.stale_after_ms * 1e6)
                    )
                    writer.append(
                        simulation.record_frame(
                            time.monotonic_ns(),
                            latest.hands,
                            latest.source_timestamp_ns,
                            latest.timestamp_ns,
                            latest.sequence_id,
                            latest.tracking_epoch,
                            latest_scale,
                            validity,
                        )
                    )
                next_tick += 1.0 / teleop_config.control_hz
                delay = next_tick - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
                else:
                    next_tick = time.monotonic()
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            if writer is not None:
                if failed or not writer.frame_count:
                    writer.abort()
                else:
                    path = writer.finish()
                    print(f"published {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arm-model", type=Path, required=True)
    parser.add_argument(
        "--urdf", type=Path, default=root / "assets/tianji_wuji2/tianji_wuji2.urdf"
    )
    parser.add_argument(
        "--left-hand-config", type=Path, default=root / "spd_vr/config/wuji2_pico_left.yaml"
    )
    parser.add_argument(
        "--right-hand-config", type=Path, default=root / "spd_vr/config/wuji2_pico_right.yaml"
    )
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--object-body",
        action="append",
        default=[],
        help="MuJoCo task-object body to include in object/contact records (repeatable)",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="deterministic spd-scene JSON; supplies object bodies and is stored in the episode manifest",
    )
    parser.add_argument("--record-to", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--collection-run-id", help="formal collection run identifier")
    parser.add_argument("--operator-id", help="formal collection operator identifier")
    parser.add_argument("--pico-serial", help="PICO device serial for the collection ledger")
    parser.add_argument("--collection-plan", type=Path, help="approved SPD-VR collection plan JSON")
    parser.add_argument("--episode-id", help="episode ID from --collection-plan")
    parser.add_argument(
        "--require-usable-training",
        action="store_true",
        help="fail closed before publishing when no complete contact-eligible SPD training window exists",
    )
    parser.add_argument("--viewer", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
