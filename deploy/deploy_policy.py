"""Deploy a policy for inference.

Launches (as multiprocessing children, see deploy/robot/launch.py):
  1. Policy server — serves inference over websocket
  2. Robot followers (one per arm) — unless --debug
  3. Camera nodes — publish frames over local IPC
  4. Policy rollout loop — sends observations, executes actions
  5. Optionally the recorder (--record)

Usage:
  # Real deployment
  uv run deploy/deploy_policy.py --checkpoint-path path/to/ckpt.pt --prompt "throw bottles"
  uv run deploy/deploy_policy.py --checkpoint-path path/to/ckpt.pt --rtc --debug

"""

import os
from pathlib import Path

import tyro

from deploy.deploy_config import DEFAULT_PROMPT, MODEL_SIZES, DeployConfig
from deploy.policy.selector import sniff_policy_kind
from deploy.robot import launch
from deploy.robot.config import get_i2rt_config
from deploy.robot.key_listeners.key_listener_config import KeyListenerConfig
from deploy.robot.launch import ProcessSpec
from deploy.robot.recorders.inference_recorder_config import InferenceRecorderConfig
from deploy.robot.specs import camera_specs, follower_specs
from deploy.robot.tasks import get_data_dir, select_task, task_to_collection_name

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Node specs
# ---------------------------------------------------------------------------


def _server_specs(cfg: DeployConfig) -> list[ProcessSpec]:
    if cfg.remote_host:
        print(
            f"[deploy_policy] Remote mode: policy server expected at "
            f"{cfg.remote_host}:{cfg.port}"
        )
        if cfg.rtc and not cfg.compress_images:
            budget_ms = cfg.rtc_inference_lead_steps * 33.33
            print(
                "[deploy_policy] WARNING: remote RTC without --compress-images "
                "sends raw camera frames over the network "
                f"(nominal inference budget ~{budget_ms:.0f}ms). If you see "
                "'inference still pending' stalls, enable --compress-images or "
                "increase RTC lead steps."
            )
        return []

    if not cfg.checkpoint_path:
        raise ValueError("--checkpoint-path is required for local deployment")
    return [
        ProcessSpec(
            "policy_server", "deploy.serve_policy:main", {"args": cfg.serve_args()}
        )
    ]


def _recorder_spec(
    cfg: DeployConfig, *, task_name: str, session_tag: str, dagger: bool = False
) -> ProcessSpec:
    rc = InferenceRecorderConfig(
        collection_name=cfg.collection_name,
        data_root_directory=cfg.data_root_directory,
        checkpoint_path=cfg.checkpoint_path or "",
        model_size=cfg.model_size,
        diffusion_steps=cfg.diffusion_steps,
        task_name=task_name,
        session_tag=session_tag,
        rtc=cfg.rtc,
        rtc_prefix_length=cfg.rtc_prefix_length if cfg.rtc else 0,
        rtc_inference_lead_steps=cfg.rtc_inference_lead_steps if cfg.rtc else 0,
        dagger=dagger,
    )
    return ProcessSpec(
        "inference_recorder",
        "deploy.robot.recorders.inference_recorder",
        {"cfg": rc},
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _real_robot_specs(
    cfg: DeployConfig, *, task_name: str, session_tag: str
) -> list[ProcessSpec]:
    specs = _server_specs(cfg)
    profile = get_i2rt_config()
    if not cfg.debug:
        specs.extend(follower_specs(profile, quiet=not cfg.verbose))
    specs.extend(camera_specs(profile))
    specs.append(
        ProcessSpec(
            "policy_rollout",
            "deploy.robot.gym.policy_rollout",
            {
                "args": cfg.rollout_config(
                    recorder_control=cfg.record,
                    direct_episode_keys=cfg.episode_control and not cfg.record,
                )
            },
            # multiprocessing replaces child stdin with /dev/null.  The rollout
            # needs the terminal for its first-action delta safety check.
            terminal_input=True,
        )
    )
    specs.append(
        ProcessSpec(
            "key_listener",
            "deploy.robot.key_listeners.key_listener",
            {
                "cfg": KeyListenerConfig(
                    name="KeyListener",
                    control_rate=60,
                )
            },
            # Key events drive the recorder (--record) or the rollout's own
            # episode control (--no-record with --episode-control). Keep the
            # listener detached only when neither consumes keys, so it cannot
            # race the first-action confirmation for terminal input.
            terminal_input=cfg.record or cfg.episode_control,
        )
    )
    if cfg.record:
        specs.append(_recorder_spec(cfg, task_name=task_name, session_tag=session_tag))
    return specs


def _configure_process(cfg: DeployConfig) -> None:
    """Set the small amount of process-wide state inherited by child nodes."""
    environment = {
        "DEPLOY_VERBOSE": "1" if cfg.verbose else None,
        "DEPLOY_INIT_Q": cfg.init_q or None,
        "DEPLOY_POST_VIDEO": None if cfg.post_video else "0",
    }
    for key, value in environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.chdir(REPO_ROOT)


def _prepare_task(cfg: DeployConfig) -> tuple[str, str]:
    """Resolve the prompt and recording metadata for a robot rollout."""
    prompt_provided = cfg.prompt is not None
    cfg.prompt = cfg.prompt or DEFAULT_PROMPT
    if not cfg.record:
        return cfg.prompt, ""

    task_name = cfg.prompt if prompt_provided else select_task()
    cfg.prompt = task_name
    if not prompt_provided or cfg.collection_name == "policy_rollout":
        cfg.collection_name = task_to_collection_name(task_name)
    cfg.data_root_directory = get_data_dir("inference_h5", cfg.data_root_directory)
    return task_name, cfg.session_tag.strip().replace(" ", "_")


def _resolve_policy(cfg: DeployConfig) -> None:
    """Detect the checkpoint kind here so the server and the recorder label agree."""
    if cfg.policy_type == "auto" and Path(cfg.checkpoint_path).expanduser().is_file():
        cfg.policy_type = sniff_policy_kind(cfg.checkpoint_path)
    if cfg.policy_type == "spd":
        raise ValueError("Tianji SPD checkpoints are simulation-only here; do not launch the YAM hardware controller")
    cfg.model_size = cfg.model_size or MODEL_SIZES.get(cfg.policy_type, "")


def main(cfg: DeployConfig) -> int:
    _configure_process(cfg)
    _resolve_policy(cfg)
    task_name, session_tag = _prepare_task(cfg)
    specs = _real_robot_specs(cfg, task_name=task_name, session_tag=session_tag)
    return launch.launch(specs)


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(DeployConfig)))
