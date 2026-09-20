"""Convert a trusted legacy SPD training checkpoint to ABC weights-only format."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abc_minimal.spd_conversion import convert_spd_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True, help="Trusted spd-paired-kv-v2 training checkpoint (pickle).")
    parser.add_argument("--output-path", type=Path, required=True, help="New checkpoint path; existing files are never overwritten.")
    parser.add_argument("--dino-checkpoint", type=Path, required=True, help="Official DINOv3 ViT-B/16 model.safetensors with adjacent config.json.")
    args = parser.parse_args()
    report = convert_spd_checkpoint(args.source_path, args.output_path, args.dino_checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
