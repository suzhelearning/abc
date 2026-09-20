"""Serve a released ABC-DiT or VLA policy over websocket."""

import logging
import os

import tyro

from deploy.policy import Policy
from deploy.policy.selector import sniff_policy_kind
from deploy.policy.vla_policy import Policy as VLAPolicy
from deploy.serve_policy_config import Args
from deploy.websocket_server import WebsocketPolicyServer


def create_policy(args: Args):
    if not args.policy.checkpoint_path:
        raise ValueError("A checkpoint is required")
    kind = args.policy_type
    if kind == "auto":
        kind = sniff_policy_kind(args.policy.checkpoint_path)
    if kind == "spd":
        raise ValueError("SPD is supported by Tianji simulation eval_policy.py, not the YAM hardware server")
    if kind == "vla":
        return VLAPolicy(args.vla_config())
    return Policy(args.dit_config())


def main(args: Args) -> None:
    level = logging.INFO if os.environ.get("DEPLOY_VERBOSE") else logging.WARNING
    logging.basicConfig(level=level, force=True)
    policy = create_policy(args)
    print(
        f"[serve_policy] steps={args.policy.diffusion_steps} "
        f"fast={args.policy.fast_inference} chunk_len={policy.chunk_len}"
    )
    print(f"[serve_policy] serving on 0.0.0.0:{args.port}")
    WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
