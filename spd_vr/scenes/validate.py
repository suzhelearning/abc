"""Validate every public SPD task over deterministic reset seeds."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

from .registry import TABLE2_STATS, TASKS


def validate_tasks(seed_start: int, seed_count: int) -> dict[str, Any]:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    results = []
    for spec in TASKS:
        expected_duration = 60.0 * spec.table2_minutes / spec.table2_episodes
        if not math.isclose(spec.target_duration_s, expected_duration, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"duration formula mismatch: {spec.qualified_name}")
        for seed in range(seed_start, seed_start + seed_count):
            first = spec.reset(seed)
            second = spec.reset(seed)
            if first.manifest() != second.manifest():
                raise AssertionError(f"reset is not deterministic: {spec.qualified_name} seed={seed}")
            ids = [item.instance_id for item in first.objects]
            if len(ids) != len(set(ids)):
                raise AssertionError(f"duplicate instance ID: {spec.qualified_name} seed={seed}")
            for item in first.objects:
                if not all(math.isfinite(float(value)) for value in (*item.position, *item.size, item.mass_kg, item.friction, *item.color_rgb)):
                    raise AssertionError(f"non-finite sampled value: {spec.qualified_name} seed={seed}")
            results.append({"task": spec.qualified_name, "seed": seed, "objects": len(first.objects), "candidate": first.candidate})
    return {
        "scenes": 6,
        "tasks": len(TASKS),
        "duration_formula": "60 * table2_minutes / table2_episodes",
        "seed_start": seed_start,
        "seed_count": seed_count,
        "resets": len(results),
        "tasks_checked": [spec.qualified_name for spec in TASKS],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args(argv)
    print(json.dumps(validate_tasks(args.seed_start, args.seed_count), sort_keys=True))
    return 0


__all__ = ["main", "validate_tasks"]

if __name__ == "__main__":
    raise SystemExit(main())
