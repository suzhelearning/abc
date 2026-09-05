"""Command line entry point for authoritative URDF model compilation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import compile_models, verify_artifacts, verify_contact_qualified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="authoritative URDF")
    parser.add_argument("--output", type=Path, required=True, help="published generated artifact directory")
    parser.add_argument("--cache", type=Path, help="collision decomposition cache directory")
    parser.add_argument("--raw-collisions", action="store_true", help="reuse source meshes without decomposition")
    parser.add_argument("--verify", action="store_true", help="verify an existing output instead of compiling")
    parser.add_argument(
        "--verify-contact",
        action="store_true",
        help="require every collision record to pass the contact surface gate",
    )
    args = parser.parse_args(argv)
    if args.verify_contact and not args.verify:
        parser.error("--verify-contact requires --verify")
    if args.verify:
        verified = verify_artifacts(args.output / "model_manifest.yaml", args.urdf)
        if args.verify_contact:
            verify_contact_qualified(verified.output_dir / "collision_manifest.yaml", urdf_path=args.urdf)
        print(f"verified {verified.full_model} and {verified.arm_model}")
    else:
        manifest = compile_models(args.urdf, args.output, args.cache, raw_collisions=args.raw_collisions)
        print(f"generated {manifest.output_dir / 'unified_plant.xml'}")
        print(f"generated {manifest.output_dir / 'arm_ik.xml'}")
        print(f"generated {manifest.path}")
        print(f"generated {manifest.collision_manifest}")
        print(f"generated {manifest.actuator_calibration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
