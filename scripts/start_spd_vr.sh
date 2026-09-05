#!/usr/bin/env bash
set -euo pipefail

# Own exactly one simulation-only process graph.  The viewer is the Zenoh
# router and MuJoCo plant owner; arm_ik and pxrea_bridge are client processes.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
session_name="spd-vr"
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
scene_manifest=""
router_timeout=300
mode="detach"
action="start"
dry_run=0
run_preflight=0

usage() {
  cat <<'EOF'
Usage: scripts/start_spd_vr.sh [options]

Actions:
  --status                 list windows in the managed session
  --stop                   stop only the managed tmux session
  --dry-run                print the three commands without starting tmux

Start options:
  --attach | --detach      attach after start, or return immediately (default)
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
  --router-timeout SEC     wait for the viewer Zenoh router before clients (default 300)
  --preflight              run read-only PICO, dependency, artifact, and port checks before tmux
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
    --attach) mode="attach"; shift ;;
    --detach) mode="detach"; shift ;;
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

if [[ "$action" == "status" ]]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux list-windows -t "$session_name" -F '#{window_name}:#{pane_current_command}'
    exit 0
  fi
  echo "SPD-VR session is not running"
  exit 1
fi

if [[ "$action" == "stop" ]]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  if tmux has-session -t "$session_name" 2>/dev/null; then
    # The target is a fixed, launcher-owned session; no process-name killing
    # or broad cleanup is performed.
    tmux kill-session -t "$session_name"
    echo "Stopped SPD-VR session: $session_name"
  else
    echo "SPD-VR session is not running"
  fi
  exit 0
fi

q() { printf '%q' "$1"; }
viewer_command=(uv run python -m spd_vr.viewer --model "$model" --urdf "$urdf" \
  --left-hand-config "$left_hand_config" --right-hand-config "$right_hand_config" \
  --endpoint "$endpoint" --listen)
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
viewer_line="$(printf '%q ' "${viewer_command[@]}")"
arm_line="$(printf '%q ' "${arm_command[@]}")"
bridge_line="$(printf '%q ' "${bridge_command[@]}")"

if ((dry_run)); then
  printf 'session=%s\n' "$session_name"
  printf 'viewer: %s\n' "$viewer_line"
  printf 'arm_ik: sleep 1 && %s\n' "$arm_line"
  printf 'pxrea_bridge: sleep 1 && %s\n' "$bridge_line"
  exit 0
fi

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
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
if ((run_preflight)); then
  preflight_command=(uv run spd-vr-preflight --repo-root "$repo_root" \
    --manifest "$(dirname "$model")/model_manifest.yaml" --urdf "$urdf" \
    --endpoint "$endpoint" --session "$session_name")
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
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "refusing duplicate session: $session_name" >&2
  exit 1
fi

tmux new-session -d -s "$session_name" -n viewer
tmux set-option -t "$session_name" remain-on-exit on >/dev/null
tmux send-keys -t "$session_name:viewer" \
  "cd $(q "$repo_root") && exec $viewer_line" C-m

wait_for_router() {
  local address host port deadline pane_dead
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
    pane_dead="$(tmux display-message -p -t "$session_name:viewer" '#{pane_dead}' 2>/dev/null || printf '1')"
    if [[ "$pane_dead" == "1" ]]; then
      echo "viewer exited before Zenoh router became ready" >&2
      tmux capture-pane -t "$session_name:viewer" -p -S -80 >&2 || true
      return 1
    fi
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      exec 3>&-
      return 0
    fi
    sleep 1
  done
  echo "timed out after ${router_timeout}s waiting for Zenoh router at $endpoint" >&2
  tmux capture-pane -t "$session_name:viewer" -p -S -80 >&2 || true
  return 1
}

if ! wait_for_router; then
  tmux kill-session -t "$session_name" 2>/dev/null || true
  exit 1
fi
tmux new-window -t "$session_name" -n arm_ik
tmux send-keys -t "$session_name:arm_ik" \
  "cd $(q "$repo_root") && sleep 1 && exec $arm_line" C-m
tmux new-window -t "$session_name" -n pxrea_bridge
tmux send-keys -t "$session_name:pxrea_bridge" \
  "cd $(q "$repo_root") && sleep 1 && exec $bridge_line" C-m
tmux select-window -t "$session_name:viewer"
echo "Started SPD-VR session: $session_name"
if [[ "$mode" == "attach" ]]; then
  exec tmux attach-session -t "$session_name"
fi
