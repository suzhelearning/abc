#!/usr/bin/env bash
set -euo pipefail

# Own exactly one foreground simulation-only process graph.  The viewer is the
# Zenoh router and MuJoCo plant owner; arm_ik and pxrea_bridge are child
# processes managed directly by this supervisor (no tmux or external daemon).
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"
endpoint="tcp/127.0.0.1:7447"
model="$repo_root/generated/spd_vr/unified_plant.xml"
arm_model="$repo_root/generated/spd_vr/arm_ik.xml"
urdf="$repo_root/assets/tianji_wuji2/tianji_wuji2.urdf"
left_hand_config="$repo_root/spd_vr/config/wuji2_pico_left.yaml"
right_hand_config="$repo_root/spd_vr/config/wuji2_pico_right.yaml"
sdk_library="${PXREA_SDK_LIBRARY:-${PXREA_SDK_ROOT:-/opt/apps/roboticsservice/SDK}/x64/libPXREARobotSDK.so}"
serial=""
fake_source=""
wait_for_shutdown=0
record_to=""
require_usable_training=0
collection_run_id=""
operator_id=""
pico_serial=""
collection_plan=""
episode_id=""
scene_manifest=""
router_timeout=300
action="start"
dry_run=0
run_preflight=0
show_viewer=0
runtime_root="${XDG_RUNTIME_DIR:-/tmp}"
runtime_dir="$runtime_root/spd-vr-${UID}"
pid_file="$runtime_dir/supervisor.pid"
lock_file="$runtime_dir/supervisor.lock"

usage() {
  cat <<'EOF'
Usage: scripts/start_spd_vr.sh [options]

Runs viewer, arm_ik, and pxrea_bridge under one foreground supervisor.
Press Ctrl-C to request a graceful stop of the complete process graph.

Actions:
  --status                 report the foreground supervisor state
  --stop                   request a graceful supervisor stop
  --dry-run                print the three child commands without starting them

Start options:
  --viewer                 show the interactive MuJoCo window
  --endpoint ENDPOINT      Zenoh endpoint (default tcp/127.0.0.1:7447)
  --model PATH             complete MuJoCo model
  --arm-model PATH         arm-only IK model
  --urdf PATH              authoritative Tianji-Wuji2 URDF
  --sdk-library PATH       PXREA shared library
  --serial ID               select one PICO/PXREA device when more than one is online
  --fake-source-jsonl PATH replay callback frames instead of loading PXREA
  --wait-for-shutdown       keep a fake source alive until spd-control shutdown
  --scene-manifest PATH    deterministic SPD scene JSON
  --record-to PATH         atomic HDF5 episode output
  --require-usable-training fail closed if the episode has no complete SPD training window
  --collection-run-id ID   formal collection run identifier (requires the next two fields)
  --operator-id ID         formal collection operator identifier
  --pico-serial ID         PICO serial for the collection ledger
  --collection-plan PATH   reviewed deterministic collection plan JSON
  --episode-id ID          planned episode ID from --collection-plan
  --router-timeout SEC     wait for the viewer Zenoh router before clients (default 300)
  --preflight              run read-only PICO, dependency, artifact, port, and supervisor checks
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --status) action="status"; shift ;;
    --stop) action="stop"; shift ;;
    --dry-run) dry_run=1; shift ;;
    --viewer) show_viewer=1; shift ;;
    --endpoint) need_value "$@"; endpoint="$2"; shift 2 ;;
    --model) need_value "$@"; model="$2"; shift 2 ;;
    --arm-model) need_value "$@"; arm_model="$2"; shift 2 ;;
    --urdf) need_value "$@"; urdf="$2"; shift 2 ;;
    --sdk-library) need_value "$@"; sdk_library="$2"; shift 2 ;;
    --serial) need_value "$@"; serial="$2"; shift 2 ;;
    --fake-source-jsonl) need_value "$@"; fake_source="$2"; shift 2 ;;
    --wait-for-shutdown) wait_for_shutdown=1; shift ;;
    --scene-manifest) need_value "$@"; scene_manifest="$2"; shift 2 ;;
    --record-to) need_value "$@"; record_to="$2"; shift 2 ;;
    --require-usable-training) require_usable_training=1; shift ;;
    --collection-run-id) need_value "$@"; collection_run_id="$2"; shift 2 ;;
    --operator-id) need_value "$@"; operator_id="$2"; shift 2 ;;
    --pico-serial) need_value "$@"; pico_serial="$2"; shift 2 ;;
    --collection-plan) need_value "$@"; collection_plan="$2"; shift 2 ;;
    --episode-id) need_value "$@"; episode_id="$2"; shift 2 ;;
    --router-timeout) need_value "$@"; router_timeout="$2"; shift 2 ;;
    --preflight) run_preflight=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$router_timeout" =~ ^[0-9]+$ ]] || ((router_timeout <= 0)); then
  echo "--router-timeout must be a positive integer" >&2
  exit 2
fi

collection_fields=0
[[ -n "$collection_run_id" ]] && ((collection_fields += 1))
[[ -n "$operator_id" ]] && ((collection_fields += 1))
[[ -n "$pico_serial" ]] && ((collection_fields += 1))
if ((collection_fields != 0 && collection_fields != 3)); then
  echo "--collection-run-id, --operator-id, and --pico-serial must be supplied together" >&2
  exit 2
fi
if ((collection_fields == 3)) && [[ -z "$record_to" ]]; then
  echo "collection metadata requires --record-to" >&2
  exit 2
fi
if [[ -n "$collection_plan" || -n "$episode_id" ]]; then
  if [[ -z "$collection_plan" || -z "$episode_id" ]]; then
    echo "--collection-plan and --episode-id must be supplied together" >&2
    exit 2
  fi
  if [[ -z "$record_to" || -z "$scene_manifest" ]]; then
    echo "--collection-plan requires --record-to and --scene-manifest" >&2
    exit 2
  fi
fi

read_supervisor_pid() {
  local value
  [[ -f "$pid_file" ]] || return 1
  IFS= read -r value <"$pid_file" || return 1
  [[ "$value" =~ ^[0-9]+$ ]] && ((value > 1)) || return 1
  printf '%s' "$value"
}

supervisor_matches() {
  local process_id="$1" command_line
  [[ -r "/proc/$process_id/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/$process_id/cmdline")"
  [[ "$command_line" == *"start_spd_vr.sh"* ]]
}

if [[ "$action" == "status" ]]; then
  supervisor_pid="$(read_supervisor_pid || true)"
  if [[ -n "$supervisor_pid" ]] && kill -0 "$supervisor_pid" 2>/dev/null \
      && supervisor_matches "$supervisor_pid"; then
    echo "SPD-VR foreground supervisor is running: pid=$supervisor_pid"
    exit 0
  fi
  echo "SPD-VR foreground supervisor is not running"
  exit 1
fi

if [[ "$action" == "stop" ]]; then
  supervisor_pid="$(read_supervisor_pid || true)"
  if [[ -z "$supervisor_pid" ]] || ! kill -0 "$supervisor_pid" 2>/dev/null; then
    echo "SPD-VR foreground supervisor is not running"
    exit 0
  fi
  if ! supervisor_matches "$supervisor_pid"; then
    echo "refusing to signal unrelated pid from $pid_file: $supervisor_pid" >&2
    exit 1
  fi
  kill -TERM "$supervisor_pid"
  for _ in {1..100}; do
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
      echo "Stopped SPD-VR foreground supervisor: pid=$supervisor_pid"
      exit 0
    fi
    sleep 0.1
  done
  echo "supervisor did not stop within 10 seconds: pid=$supervisor_pid" >&2
  exit 1
fi

q() { printf '%q' "$1"; }
viewer_command=(uv run python -m spd_vr.viewer --model "$model" --urdf "$urdf" \
  --left-hand-config "$left_hand_config" --right-hand-config "$right_hand_config" \
  --endpoint "$endpoint" --listen)
if ((show_viewer)); then
  viewer_command+=(--viewer)
fi
arm_command=(uv run python -m spd_vr.arm_ik --model "$arm_model" --urdf "$urdf" \
  --endpoint "$endpoint")
if [[ -n "$fake_source" ]]; then
  bridge_command=(uv run python -m spd_vr.pxrea_bridge --fake-source-jsonl "$fake_source" \
    --endpoint "$endpoint")
  if ((wait_for_shutdown)); then
    bridge_command+=(--wait-for-shutdown)
  fi
else
  if ((wait_for_shutdown)); then
    echo "--wait-for-shutdown is only valid with --fake-source-jsonl" >&2
    exit 2
  fi
  bridge_command=(uv run python -m spd_vr.pxrea_bridge --sdk-library "$sdk_library" \
    --endpoint "$endpoint")
fi
if [[ -n "$serial" ]]; then
  bridge_command+=(--device-id "$serial")
fi
if [[ -n "$scene_manifest" ]]; then
  viewer_command+=(--scene-manifest "$scene_manifest")
fi
if [[ -n "$collection_plan" ]]; then
  viewer_command+=(--collection-plan "$collection_plan" --episode-id "$episode_id")
fi
if [[ -n "$record_to" ]]; then
  viewer_command+=(--record-to "$record_to")
fi
if ((require_usable_training)); then
  if [[ -z "$record_to" ]]; then
    echo "--require-usable-training requires --record-to" >&2
    exit 2
  fi
  viewer_command+=(--require-usable-training)
fi
if ((collection_fields == 3)); then
  viewer_command+=(--collection-run-id "$collection_run_id" --operator-id "$operator_id" --pico-serial "$pico_serial")
fi
viewer_line="$(printf '%q ' "${viewer_command[@]}")"
arm_line="$(printf '%q ' "${arm_command[@]}")"
bridge_line="$(printf '%q ' "${bridge_command[@]}")"

if ((dry_run)); then
  printf 'mode=foreground\n'
  printf 'viewer: %s\n' "$viewer_line"
  printf 'arm_ik: %s\n' "$arm_line"
  printf 'pxrea_bridge: %s\n' "$bridge_line"
  exit 0
fi

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
required_files=("$model" "$arm_model" "$urdf" "$left_hand_config" "$right_hand_config")
if [[ -n "$fake_source" ]]; then
  required_files+=("$fake_source")
else
  required_files+=("$sdk_library")
fi
for required in "${required_files[@]}"; do
  [[ -f "$required" ]] || { echo "required file is missing: $required" >&2; exit 1; }
done
if [[ -n "$scene_manifest" ]]; then
  [[ -f "$scene_manifest" ]] || { echo "scene manifest is missing: $scene_manifest" >&2; exit 1; }
fi
if [[ -n "$collection_plan" ]]; then
  [[ -f "$collection_plan" ]] || { echo "collection plan is missing: $collection_plan" >&2; exit 1; }
fi
mkdir -p "$runtime_dir"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "refusing duplicate foreground supervisor: $lock_file is locked" >&2
  exit 1
fi

child_pids=()
cleanup() {
  local exit_status=$? recorded_pid="" process_id
  trap - EXIT INT TERM
  for process_id in "${child_pids[@]}"; do
    kill -TERM "$process_id" 2>/dev/null || true
  done
  for process_id in "${child_pids[@]}"; do
    wait "$process_id" 2>/dev/null || true
  done
  if [[ -f "$pid_file" ]]; then
    IFS= read -r recorded_pid <"$pid_file" || true
    if [[ "$recorded_pid" == "$$" ]]; then
      rm -f -- "$pid_file"
    fi
  fi
  exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ((run_preflight)); then
  preflight_command=(uv run spd-vr-preflight --repo-root "$repo_root" \
    --manifest "$(dirname "$model")/model_manifest.yaml" --urdf "$urdf" \
    --endpoint "$endpoint" --supervisor-pid-file "$pid_file")
  if [[ -n "$fake_source" ]]; then
    preflight_command+=(--fake-source "$fake_source")
  else
    preflight_command+=(--sdk-library "$sdk_library")
  fi
  if [[ -n "$serial" ]]; then
    preflight_command+=(--serial "$serial")
  fi
  if [[ -n "$record_to" ]]; then
    preflight_command+=(--require-contact)
  fi
  "${preflight_command[@]}"
fi
printf '%s\n' "$$" >"$pid_file"

PYTHONUNBUFFERED=1 "${viewer_command[@]}" \
  > >(sed -u 's/^/[viewer] /') 2>&1 &
viewer_pid=$!
child_pids+=("$viewer_pid")

wait_for_router() {
  local address host port deadline
  case "$endpoint" in
    tcp/\[*\]:*)
      address="${endpoint#tcp/}"
      host="${address%%]:*}]"
      host="${host#[}"
      port="${address##*:}"
      ;;
    tcp/*:*)
      address="${endpoint#tcp/}"
      host="${address%:*}"
      port="${address##*:}"
      ;;
    *)
      echo "cannot wait for unsupported Zenoh endpoint: $endpoint" >&2
      return 2
      ;;
  esac
  deadline=$((SECONDS + router_timeout))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$viewer_pid" 2>/dev/null; then
      echo "viewer exited before Zenoh router became ready" >&2
      return 1
    fi
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      exec 3>&-
      return 0
    fi
    sleep 1
  done
  echo "timed out after ${router_timeout}s waiting for Zenoh router at $endpoint" >&2
  return 1
}

if ! wait_for_router; then
  exit 1
fi

PYTHONUNBUFFERED=1 "${arm_command[@]}" \
  > >(sed -u 's/^/[arm_ik] /') 2>&1 &
arm_pid=$!
child_pids+=("$arm_pid")

PYTHONUNBUFFERED=1 "${bridge_command[@]}" \
  > >(sed -u 's/^/[pxrea_bridge] /') 2>&1 &
bridge_pid=$!
child_pids+=("$bridge_pid")

declare -A child_names=(
  ["$viewer_pid"]="viewer"
  ["$arm_pid"]="arm_ik"
  ["$bridge_pid"]="pxrea_bridge"
)
active_pids=("${child_pids[@]}")

echo "Started SPD-VR foreground supervisor: pid=$$"
echo "  viewer=$viewer_pid arm_ik=$arm_pid pxrea_bridge=$bridge_pid"
echo "Press Ctrl-C for a graceful stop; use spd-control --pedal in terminal 2."

while ((${#active_pids[@]})); do
  finished_pid=""
  child_status=0
  wait -n -p finished_pid "${active_pids[@]}" || child_status=$?
  if [[ -z "$finished_pid" ]]; then
    echo "supervisor could not identify the exited child process" >&2
    exit 1
  fi
  child_name="${child_names[$finished_pid]:-unknown}"
  remaining_pids=()
  for process_id in "${active_pids[@]}"; do
    if [[ "$process_id" != "$finished_pid" ]]; then
      remaining_pids+=("$process_id")
    fi
  done
  active_pids=("${remaining_pids[@]}")
  echo "$child_name exited with status $child_status"
  if [[ "$child_name" == "viewer" ]]; then
    exit "$child_status"
  fi
  if ((child_status != 0)); then
    exit "$child_status"
  fi
done

echo "viewer did not remain in the managed process graph" >&2
exit 1
