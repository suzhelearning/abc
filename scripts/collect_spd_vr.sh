#!/usr/bin/env bash
set -euo pipefail

# High-level, two-terminal collection entry point.  Formal identity and file
# paths come from the reviewed collection plan; start_spd_vr.sh remains the
# lower-level foreground process supervisor.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: scripts/collect_spd_vr.sh RUN_DIR EPISODE_ID [--dry-run]

Terminal 1: run this command and keep it in the foreground.
Terminal 2: run `uv run spd-control --pedal`.

RUN_DIR must contain collection-plan.json and EPISODE_ID.scene.json. The
matching scene model is read from generated/spd_vr/EPISODE_ID.xml, and the
episode is published atomically to RUN_DIR/EPISODE_ID.hdf5.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (($# < 2 || $# > 3)); then
  usage >&2
  exit 2
fi
if (($# == 3)) && [[ "$3" != "--dry-run" ]]; then
  echo "unknown option: $3" >&2
  usage >&2
  exit 2
fi

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
run_dir_input="$1"
episode_id="$2"
[[ -d "$run_dir_input" ]] || { echo "run directory is missing: $run_dir_input" >&2; exit 1; }
run_dir="$(cd "$run_dir_input" && pwd -P)"
collection_plan="$run_dir/collection-plan.json"
[[ -f "$collection_plan" ]] || { echo "collection plan is missing: $collection_plan" >&2; exit 1; }

episode_count="$(
  jq -er --arg id "$episode_id" \
    '[.episodes[] | select(.episode_id == $id)] | length' "$collection_plan"
)"
if [[ "$episode_count" != "1" ]]; then
  echo "episode ID must occur exactly once in collection plan: $episode_id" >&2
  exit 1
fi

collection_run_id="$(jq -er '.collection_identity.run_id | select(type == "string" and length > 0)' "$collection_plan")"
operator_id="$(jq -er '.collection_identity.operator_id | select(type == "string" and length > 0)' "$collection_plan")"
pico_serial="$(jq -er '.collection_identity.pico_serial | select(type == "string" and length > 0)' "$collection_plan")"

model="$repo_root/generated/spd_vr/${episode_id}.xml"
scene_manifest="$run_dir/${episode_id}.scene.json"
record_to="$run_dir/${episode_id}.hdf5"
launcher_args=(
  --preflight
  --viewer
  --serial "$pico_serial"
  --model "$model"
  --arm-model "$repo_root/generated/spd_vr/arm_ik.xml"
  --scene-manifest "$scene_manifest"
  --record-to "$record_to"
  --collection-plan "$collection_plan"
  --episode-id "$episode_id"
  --collection-run-id "$collection_run_id"
  --operator-id "$operator_id"
  --pico-serial "$pico_serial"
  --require-usable-training
)
if (($# == 3)); then
  launcher_args+=(--dry-run)
fi

exec "$repo_root/scripts/start_spd_vr.sh" "${launcher_args[@]}"
