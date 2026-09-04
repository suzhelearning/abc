"""Compatibility entry point for the authoritative URDF model compiler.

Older SPD scripts imported ``model_builder.build_model``.  Keep that small
interface while routing all generation through ``model_compiler.artifacts``;
there is one source of truth for the full 54-DoF plant, arm model and hashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_compiler.artifacts import compile_models


def workspace_root() -> Path:
    """Return the repository root for both editable and installed execution."""
    return Path(__file__).resolve().parents[1]


def default_urdf() -> Path:
    return workspace_root() / "assets" / "tianji_wuji2" / "tianji_wuji2.urdf"


def build_model(
    output_dir: str | Path | None = None,
    *,
    urdf_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    raw_collisions: bool = False,
) -> tuple[Path, Path, Path]:
    """Compile and return ``(unified_xml, model_manifest, actuator_yaml)``."""
    output = Path(output_dir) if output_dir is not None else workspace_root() / "generated" / "spd_vr"
    urdf = Path(urdf_path) if urdf_path is not None else default_urdf()
    cache = Path(cache_dir) if cache_dir is not None else output.parent / "collision_cache"
    result = compile_models(urdf, output, cache, raw_collisions=raw_collisions)
    return result.full_model, result.path, result.actuator_calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--raw-collisions", action="store_true")
    args = parser.parse_args(argv)
    full_model, manifest, calibration = build_model(
        args.output_dir,
        urdf_path=args.urdf,
        cache_dir=args.cache,
        raw_collisions=args.raw_collisions,
    )
    print(f"generated {full_model}")
    print(f"generated {manifest}")
    print(f"generated {calibration}")
    return 0


__all__ = ["build_model", "default_urdf", "main", "workspace_root"]

if __name__ == "__main__":
    raise SystemExit(main())
