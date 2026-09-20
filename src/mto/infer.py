"""Run MTO from local assets, on an NPZ observation or as a WebSocket policy.

NPZ inputs contain head_camera, left_camera, right_camera (HWC uint8 RGB),
state (14 raw qpos values) and prompt (a scalar string). Omit observation_path
to serve the same observation contract over the existing OpenPI WebSocket API.
The checkpoint path must point to the params directory, not its parent step.
"""

import dataclasses
import time
from typing import Literal

import numpy as np
import tyro

from openpi.models.mto_config import TrainingMode, create_mto_config
from openpi.policies.mto_policy import MtoPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


@dataclasses.dataclass
class Args:
    params_path: str
    move_norm_stats_path: str
    operate_norm_stats_path: str
    tokenizer_path: str
    # Match the saved architecture; lora and lora_frozen_vlm have different trees.
    training_mode: TrainingMode = "full"
    action_horizon: int = 30
    model_action_dim: int = 32
    max_token_len: int = 50
    paligemma_variant: str | None = None
    action_expert_variant: str | None = None
    normalize_method: Literal["zscore", "min_max"] = "zscore"
    num_steps: int = 10
    seed: int = 0
    warmup: bool = True
    service_id: str | None = None
    # Local NPZ input. Omit to start the WebSocket server.
    observation_path: str | None = None
    output_path: str = "mto_actions.npz"
    host: str = "0.0.0.0"
    port: int = 8000


def main(args: Args) -> None:
    config = create_mto_config(
        args.training_mode,
        action_dim=args.model_action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_expert_variant,
    )
    policy = MtoPolicy(
        params_path=args.params_path,
        move_norm_stats_path=args.move_norm_stats_path,
        operate_norm_stats_path=args.operate_norm_stats_path,
        tokenizer_path=args.tokenizer_path,
        config=config,
        normalize_method=args.normalize_method,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    if args.observation_path is None:
        if args.warmup:
            start = time.monotonic()
            policy.warmup()
            print(f"Compiled both expert decoders in {time.monotonic() - start:.2f}s", flush=True)
        WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata={**policy.metadata, "service_id": args.service_id}).serve_forever()
    else:
        with np.load(args.observation_path) as sample:
            observation = {
                "images": {name: sample[name] for name in ("head_camera", "left_camera", "right_camera")},
                "state": sample["state"],
                "prompt": sample["prompt"].item(),
            }
        result = policy.infer(observation)
        np.savez(args.output_path, **result)
        print(f"Saved {result['phase']} action chunk to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
