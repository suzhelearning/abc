"""Run a bounded, read-only SPD-VR acceptance set.

The command deliberately does not collect data or start a process graph.  It
checks the deterministic task registry, generated model/contact gates, and an
optional directory of HDF5 episodes.  Every check is emitted as JSON so a lab
run can be archived beside its manifest; a missing requested input is a
failure, never an implicit pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import scan_episodes, validate_episode
from .model_compiler.artifacts import verify_artifacts, verify_contact_qualified
from .replay import replay_episode
from .robot import RobotSpec
from .scenes.manifest import load_scene_manifest
from .scenes.validate import validate_tasks


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """One acceptance result with JSON-safe optional metrics."""

    name: str
    ok: bool
    detail: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_object_bodies(path: Path) -> list[str]:
    """Read object body names from a scene manifest without trusting them.

    ``replay_episode`` still validates the episode/model hashes.  This helper
    only supplies the optional names needed for MuJoCo contact comparison.
    """
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            raw = handle["manifest/json"][()]
        document = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return []
    scene = document.get("scene_manifest") if isinstance(document, Mapping) else None
    names = scene.get("object_bodies") if isinstance(scene, Mapping) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    runtime = document.get("runtime") if isinstance(document, Mapping) else None
    names = runtime.get("object_bodies") if isinstance(runtime, Mapping) else None
    return list(names) if isinstance(names, list) and all(isinstance(name, str) for name in names) else []


def _check_scenes(seed_start: int, seed_count: int) -> AcceptanceResult:
    try:
        report = validate_tasks(seed_start, seed_count)
    except Exception as exc:
        return AcceptanceResult("scenes", False, str(exc))
    return AcceptanceResult(
        "scenes",
        True,
        f"validated {report['tasks']} tasks over {report['resets']} deterministic resets",
        {key: report[key] for key in ("scenes", "tasks", "resets", "seed_start", "seed_count")},
    )


def _check_scene_manifest(path: Path, *, root: Path, model: Path) -> AcceptanceResult:
    try:
        document = load_scene_manifest(path)
        scene_model = document.get("model")
        if isinstance(scene_model, Mapping) and scene_model.get("sha256") is not None:
            expected = scene_model["sha256"]
            if not isinstance(expected, str) or not model.is_file() or _sha256(model) != expected:
                raise ValueError("scene manifest model hash does not match the requested model")
        source_hashes = document.get("builder_source_sha256", {})
        required_sources = {"registry.py", "scene_builder.py", "model_scene.py"}
        if (
            not isinstance(source_hashes, Mapping)
            or not required_sources.issubset(source_hashes)
        ):
            raise ValueError("scene builder source hashes are incomplete")
        source_root = root / "spd_vr" / "scenes"
        for name, expected in source_hashes.items():
            source = source_root / str(name)
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in expected)
            ):
                raise ValueError("scene builder source hashes are malformed")
            if not source.is_file() or _sha256(source) != expected:
                raise ValueError(f"scene builder source hash mismatch: {name}")
    except Exception as exc:
        return AcceptanceResult("scene_manifest", False, str(exc))
    return AcceptanceResult(
        "scene_manifest",
        True,
        f"validated {path}",
        {"task": document["task"], "objects": len(document["object_bodies"])},
    )


def _declared_scene_model(path: Path) -> Path | None:
    """Return the model named by a valid scene manifest, if it declares one."""
    try:
        document = load_scene_manifest(path)
    except Exception:
        # The normal scene-manifest result reports the detailed validation
        # error.  Do not hide that error while resolving an optional model.
        return None
    model = document.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("path"), str):
        return None
    declared = Path(model["path"]).expanduser()
    return declared if declared.is_absolute() else (path.parent / declared)


def _check_artifacts(
    manifest_path: Path,
    urdf_path: Path,
    *,
    require_contact: bool,
) -> tuple[AcceptanceResult, AcceptanceResult | None]:
    try:
        verified = verify_artifacts(manifest_path, urdf_path)
    except Exception as exc:
        result = AcceptanceResult("artifacts", False, str(exc))
        contact = AcceptanceResult("contact_gate", False, "artifacts are not verified") if require_contact else None
        return result, contact
    artifact = AcceptanceResult(
        "artifacts",
        True,
        f"verified {verified.output_dir}",
        {
            "model": str(verified.full_model),
            "arm_model": str(verified.arm_model),
            "model_sha256": _sha256(verified.full_model),
        },
    )
    if not require_contact:
        return artifact, None
    try:
        report = verify_contact_qualified(verified.output_dir / "collision_manifest.yaml", urdf_path=urdf_path)
    except Exception as exc:
        return artifact, AcceptanceResult("contact_gate", False, str(exc))
    return artifact, AcceptanceResult("contact_gate", True, "all collision records pass surface gate", report)


def _check_episode(path: Path, *, verify_checksums: bool) -> AcceptanceResult:
    try:
        manifest = validate_episode(path, verify_checksums=verify_checksums)
    except Exception as exc:
        return AcceptanceResult("episode", False, f"{path}: {exc}")
    return AcceptanceResult(
        "episode",
        True,
        f"validated {path}",
        {
            "path": str(path),
            "raw_frames": int(manifest.get("raw_frames", 0)),
            "training_frames": int(manifest.get("training_frames", 0)),
            "checksum_verified": bool(verify_checksums),
        },
    )


def _check_replay(
    path: Path,
    *,
    model_path: Path,
    urdf_path: Path,
    tolerance: float,
    render: bool,
) -> AcceptanceResult:
    try:
        from .simulation import SPDVRSim

        robot = RobotSpec.from_urdf(urdf_path)
        with SPDVRSim(
            model_path,
            robot,
            backend="mujoco",
            object_bodies=_episode_object_bodies(path),
        ) as simulator:
            report = replay_episode(
                path,
                simulator=simulator,
                model_path=model_path,
                tolerance=tolerance,
                render=render,
            )
        if not report.get("valid", False):
            return AcceptanceResult("replay", False, f"replay mismatches in {path}", report)
        return AcceptanceResult("replay", True, f"replayed {path}", report)
    except Exception as exc:
        return AcceptanceResult("replay", False, f"{path}: {exc}")


def run_acceptance(
    *,
    repo_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    episodes_path: str | Path | None = None,
    scene_manifest: str | Path | None = None,
    model_path: str | Path | None = None,
    seed_start: int = 0,
    seed_count: int = 1,
    require_contact: bool = False,
    verify_checksums: bool = True,
    replay: bool = False,
    replay_tolerance: float = 1e-6,
    render: bool = False,
    max_episodes: int | None = None,
) -> list[AcceptanceResult]:
    """Run the requested acceptance checks without changing repository state."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    urdf = Path(urdf_path) if urdf_path is not None else root / "assets" / "tianji_wuji2" / "tianji_wuji2.urdf"
    manifest = Path(manifest_path) if manifest_path is not None else root / "generated" / "spd_vr" / "model_manifest.yaml"
    model = Path(model_path) if model_path is not None else manifest.parent / "unified_plant.xml"
    if model_path is None and scene_manifest is not None:
        model = _declared_scene_model(Path(scene_manifest)) or model
    results: list[AcceptanceResult] = [_check_scenes(seed_start, seed_count)]

    if scene_manifest is not None:
        results.append(_check_scene_manifest(Path(scene_manifest), root=root, model=model))

    artifact, contact = _check_artifacts(manifest, urdf, require_contact=require_contact)
    results.append(artifact)
    if contact is not None:
        results.append(contact)

    if episodes_path is None:
        results.append(AcceptanceResult("episodes", True, "not requested; pass --episodes for HDF5 acceptance"))
        return results
    episodes = scan_episodes(episodes_path)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[:max_episodes]
    if not episodes:
        results.append(AcceptanceResult("episodes", False, f"no HDF5 episodes found under {episodes_path}"))
        return results
    episode_results = [_check_episode(path, verify_checksums=verify_checksums) for path in episodes]
    failed = [result for result in episode_results if not result.ok]
    results.append(
        AcceptanceResult(
            "episodes",
            not failed,
            f"validated {len(episodes) - len(failed)}/{len(episodes)} HDF5 episodes",
            {"count": len(episodes), "failed": len(failed)},
        )
    )
    results.extend(episode_results)
    if replay:
        if not artifact.ok:
            results.append(AcceptanceResult("replay", False, "replay requires verified model artifacts"))
        elif not model.is_file():
            results.append(AcceptanceResult("replay", False, f"missing replay model: {model}"))
        else:
            results.extend(
                _check_replay(
                    path,
                    model_path=model,
                    urdf_path=urdf,
                    tolerance=replay_tolerance,
                    render=render,
                )
                for path in episodes
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=None, help="episode file or directory to validate")
    parser.add_argument("--scene-manifest", type=Path, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--require-contact", action="store_true")
    parser.add_argument("--no-checksums", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--replay-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        results = run_acceptance(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            model_path=args.model,
            urdf_path=args.urdf,
            episodes_path=args.episodes,
            scene_manifest=args.scene_manifest,
            seed_start=args.seed_start,
            seed_count=args.seed_count,
            require_contact=args.require_contact,
            verify_checksums=not args.no_checksums,
            replay=args.replay,
            replay_tolerance=args.replay_tolerance,
            render=args.render,
            max_episodes=args.max_episodes,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    report = {
        "ok": all(result.ok for result in results),
        "results": [result.as_dict() for result in results],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


__all__ = ["AcceptanceResult", "main", "run_acceptance"]

if __name__ == "__main__":
    raise SystemExit(main())
