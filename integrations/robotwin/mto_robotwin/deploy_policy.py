"""Three RGB cameras and raw joint14 in; absolute qpos actions out.

The runner sets ``episode_seed`` to the accepted scene seed before reset_model.
This adapter has no access to demonstration phases or future observations.
"""

import json
from pathlib import Path
import time

import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class MtoRoboTwinPolicy:
    def __init__(self, args):
        self.client = WebsocketClientPolicy(args.get("server_host", "127.0.0.1"), int(args.get("server_port", 9700)))
        self.metadata = self.client.get_server_metadata()
        self.exec_steps = int(args.get("exec_steps", 30))
        assert 1 <= self.exec_steps <= self.metadata["action_horizon"]
        self.trace_dir = Path(args.get("trace_dir", "mto_traces"))
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.episode_seed = None
        self.episode_index = -1
        self.stage = "model_connect"

    def write_trace(self, record):
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


def get_model(usr_args):
    return MtoRoboTwinPolicy(usr_args)


def reset_model(model):
    model.stage = "model_reset"
    model.client.reset(seed=model.episode_seed)
    model.episode_index += 1
    model.chunk_count = 0
    model.route_counts = {"move": 0, "operate": 0}
    model.route_switches = 0
    model.last_phase = None
    model.inference_seconds = []
    model.gripper_clipped_values = 0
    identity = f"seed_{model.episode_seed}" if model.episode_seed is not None else f"episode_{model.episode_index}"
    model.trace_path = model.trace_dir / f"{identity}.jsonl"
    model.trace_path.write_text("", encoding="utf-8")
    model.write_trace({"event": "reset", "seed": model.episode_seed, "exec_steps": model.exec_steps,
                       "server_metadata": model.metadata})


def eval(TASK_ENV, model, observation):
    if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
        return
    model.stage = "environment_observation"
    images = {name: np.asarray(observation["observation"][name]["rgb"])
              for name in ("head_camera", "left_camera", "right_camera")}
    for image in images.values():
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    assert state.shape == (14,) and np.isfinite(state).all()
    prompt = TASK_ENV.get_instruction()
    model.stage = "model_inference"
    started = time.perf_counter()
    result = model.client.infer({
        "images": images,
        "state": state,
        "prompt": prompt,
    })
    elapsed = time.perf_counter() - started
    actions = np.asarray(result["actions"], dtype=np.float32)
    assert actions.shape == (model.metadata["action_horizon"], 14) and np.isfinite(actions).all()
    phase = result["phase"]
    model.route_counts[phase] += 1
    model.route_switches += int(model.last_phase is not None and phase != model.last_phase)
    model.last_phase = phase
    model.inference_seconds.append(elapsed)
    model.write_trace({"event": "chunk", "chunk": model.chunk_count, "step": int(TASK_ENV.take_action_cnt),
                       "phase": phase, "route_probabilities": np.asarray(result["route_probabilities"]).tolist(),
                       "inference_seconds": elapsed, "actions": actions.tolist()})
    # Grippers are absolute continuous targets in [0, 1]; joints are not clipped.
    executed = actions[:model.exec_steps].copy()
    executed[:, [6, 13]] = np.clip(executed[:, [6, 13]], 0.0, 1.0)
    for index, action in enumerate(executed):
        model.stage = "environment_action"
        TASK_ENV.take_action(action, action_type="qpos")
        clipped = int(np.count_nonzero(action[[6, 13]] != actions[index, [6, 13]]))
        model.gripper_clipped_values += clipped
        model.write_trace({"event": "action", "chunk": model.chunk_count, "index": index,
                           "step": int(TASK_ENV.take_action_cnt), "action": action.tolist(),
                           "success": bool(TASK_ENV.eval_success), "gripper_clipped_values": clipped})
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        # RoboTwin's recorder uses now_obs during take_action. Refresh it even
        # inside an open-loop chunk; inference still runs only once per chunk.
        model.stage = "environment_observation"
        TASK_ENV.get_obs()
    model.chunk_count += 1
