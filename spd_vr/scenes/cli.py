"""Build and validate deterministic SPD-VR scene/task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .model_scene import write_scene_model
from .registry import SCENES, TASKS, get_task
from .validate import validate_tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_tasks() -> list[dict[str, object]]:
    return [
        {
            "scene": spec.scene,
            "task": spec.name,
            "qualified_name": spec.qualified_name,
            "prompt": spec.prompt,
            "target_duration_s": spec.target_duration_s,
            "table2_episodes": spec.table2_episodes,
            "table2_minutes": spec.table2_minutes,
        }
        for spec in TASKS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print all six scenes and 17 tasks")
    parser.add_argument("--scene", help="scene name, or scene/task")
    parser.add_argument("--task", help="task name when --scene is not qualified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", type=Path, help="verified unified_plant.xml")
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validate-seeds", type=int, metavar="COUNT")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps({"scenes": list(SCENES), "tasks": _list_tasks()}, indent=2, ensure_ascii=False))
        return 0
    if args.validate_seeds is not None:
        print(json.dumps(validate_tasks(args.seed_start, args.validate_seeds), sort_keys=True))
        return 0
    if not args.scene:
        parser.error("one of --list, --validate-seeds, or --scene is required")
    if args.output_model is not None and args.base_model is None:
        parser.error("--output-model requires --base-model")

    spec = get_task(args.scene, args.task)
    result = spec.reset(args.seed)
    document = {
        "schema_version": "spd-vr-scene-v1",
        "task": spec.qualified_name,
        "prompt": spec.prompt,
        "target_duration_s": spec.target_duration_s,
        "table2": {
            "episodes": spec.table2_episodes,
            "minutes": spec.table2_minutes,
        },
        "reset": result.manifest(),
        "object_bodies": [item.name for item in result.objects],
        "builder_source_sha256": {
            name: _sha256(Path(__file__).resolve().parent / name)
            for name in ("registry.py", "scene_builder.py", "model_scene.py")
        },
    }
    if args.output_model is not None:
        output_model = write_scene_model(args.base_model, result, args.output_model)
        document["model"] = {
            "path": str(output_model.resolve()),
            "sha256": _sha256(output_model),
        }
    manifest_path = args.manifest
    if manifest_path is None and args.output_model is not None:
        manifest_path = args.output_model.with_suffix(".scene.json")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        document["manifest"] = str(manifest_path.resolve())
        manifest_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
