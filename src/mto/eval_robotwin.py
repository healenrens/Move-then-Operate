"""MTO-owned sequential evaluation against the current RoboTwin environment.

Environment/configuration APIs were read at RoboTwin commit
6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755 (2026-09-20):
https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/scripts/eval_policy_xpolicylab.py

We use RoboTwin environments and its instruction generator, not its policy
runner or XPolicyLab transport. The controller runs in MTO's environment; each
task worker runs in ROBOTWIN_PY. Only normal success/step-limit termination
produces an episode result. An uncaught exception retains its traceback and
writes an attempt record, never a failed/completed policy trial.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def records(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def load_environment_args(root: Path, task: str, setting: str) -> dict:
    import yaml
    from envs import CONFIGS_PATH

    config_dir = root / "env_cfg" / "task_config"
    args = yaml.safe_load((config_dir / f"{setting}.yml").read_text())
    args.update(task_name=task, task_config=setting, policy_name="mto_robotwin", eval_mode=True)
    args["data_type"].update(rgb=True, qpos=True)
    args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
    embodiments = yaml.safe_load((Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text())
    cameras = yaml.safe_load((Path(CONFIGS_PATH) / "_camera_config.yml").read_text())
    head = cameras[args["camera"]["head_camera_type"]]
    args.update(head_camera_h=head["h"], head_camera_w=head["w"])
    names = args["embodiment"]
    args["left_robot_file"] = embodiments[names[0]]["file_path"]
    args["right_robot_file"] = embodiments[names[0] if len(names) == 1 else names[1]]["file_path"]
    args["dual_arm_embodied"] = len(names) == 1
    if len(names) == 3:
        args["embodiment_dis"] = names[2]
    for side in ("left", "right"):
        args[f"{side}_embodiment_config"] = yaml.safe_load(
            (Path(args[f"{side}_robot_file"]) / "config.yml").read_text())
    return args


def run_worker(job_path: Path) -> None:
    job = json.loads(job_path.read_text())
    root = Path(job["robotwin_root"])
    os.chdir(root)
    for path in (root, root / "description" / "utils", Path(job["mto_root"]) / "integrations" / "robotwin"):
        sys.path.insert(0, str(path))
    directory = Path(job["directory"])
    progress = {"seed": job["next_seed"], "stage": "environment_import"}
    model = None

    def record_exception(kind, error, traceback):
        stage = model.stage if progress["stage"] == "rollout" else progress["stage"]
        # These are scene-selection failures in RoboTwin's reference runner.
        candidate_error = stage in {"expert_setup", "expert_play", "expert_check", "evaluation_setup"}
        record = {
            "seed": progress["seed"], "status": "candidate_error" if candidate_error else "error",
            "stage": stage, "category": "model" if stage.startswith("model") else "environment",
            "error_type": kind.__name__, "message": str(error),
        }
        write_json(directory / "attempts" / f"seed_{progress['seed']}.json", record)
        write_json(directory / "errors" / f"seed_{progress['seed']}_{time.time_ns()}.json", record)
        sys.__excepthook__(kind, error, traceback)

    sys.excepthook = record_exception
    import numpy as np
    from generate_episode_instructions import generate_episode_descriptions
    from mto_robotwin import eval as policy_eval, get_model, reset_model

    progress["stage"] = "environment_config"
    args = load_environment_args(root, job["task"], job["task_config"])
    env_class = getattr(importlib.import_module(f"envs.{job['task']}"), job["task"])
    env = env_class()
    render_freq = args["render_freq"]
    progress["stage"] = "model_connect"
    model = get_model({**job, "trace_dir": str(directory / "traces")})
    completed = len(records(directory / "episodes"))
    env.suc = 0
    env.test_num = completed
    for seed in range(job["next_seed"], job["seed_stop"]):
        if completed >= job["test_num"]:
            break
        progress.update(seed=seed, stage="expert_setup")
        args.update(render_freq=0, eval_video_save_dir=None)
        env.setup_demo(now_ep_num=completed, seed=seed, is_test=True, **args)
        progress["stage"] = "expert_play"
        episode_info = env.play_once()
        env.close_env()
        progress["stage"] = "expert_check"
        if not (env.plan_success and env.check_success()):
            write_json(directory / "attempts" / f"seed_{seed}.json",
                       {"seed": seed, "status": "skipped", "reason": "expert_failed"})
            continue

        video_dir = directory / "videos" / f"seed_{seed}"
        video_path = video_dir / f"episode{completed}.mp4" if job["video"] else None
        if job["video"]:
            video_dir.mkdir(parents=True, exist_ok=True)
        args.update(render_freq=render_freq, eval_video_save_dir=str(video_dir) if job["video"] else None)
        progress["stage"] = "evaluation_setup"
        env.setup_demo(now_ep_num=completed, seed=seed, is_test=True, **args)
        progress["stage"] = "instruction"
        if job["deterministic_instruction"]:
            random.seed(seed)
        candidates = generate_episode_descriptions(job["task"], [episode_info["info"]], job["test_num"])
        instruction = str(np.random.choice(candidates[0][job["instruction_type"]]))
        env.set_instruction(instruction=instruction)
        if job["video"]:
            progress["stage"] = "video_start"
            ffmpeg = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{args['head_camera_w']}x{args['head_camera_h']}", "-framerate", "10",
                "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(video_path),
            ], stdin=subprocess.PIPE)
            env._set_eval_video_ffmpeg(ffmpeg)
        model.episode_seed = seed
        progress["stage"] = "rollout"
        reset_model(model)
        while not (env.eval_success or env.take_action_cnt >= env.step_lim):
            model.stage = "environment_observation"
            observation = env.get_obs()
            policy_eval(env, model, observation)
        progress["stage"] = "video_finish"
        if job["video"]:
            env._del_eval_video_ffmpeg()
        result = {
            "status": "completed", "task": job["task"], "task_config": job["task_config"],
            "trial_index": completed, "seed": seed, "success": bool(env.eval_success),
            "termination": "success" if env.eval_success else "step_limit",
            "steps": int(env.take_action_cnt), "step_limit": int(env.step_lim), "instruction": instruction,
            "instruction_type": job["instruction_type"], "exec_steps": model.exec_steps,
            "action_horizon": model.metadata["action_horizon"], "route_counts": model.route_counts,
            "route_switches": model.route_switches, "inference_seconds": model.inference_seconds,
            "gripper_clipped_values": model.gripper_clipped_values,
            "trace": str(model.trace_path), "video": str(video_path) if video_path else None,
        }
        model.write_trace({"event": "terminal", "success": result["success"], "termination": result["termination"],
                           "steps": result["steps"]})
        write_json(directory / "episodes" / f"seed_{seed}.json", result)
        write_json(directory / "attempts" / f"seed_{seed}.json", {"seed": seed, "status": "completed"})
        completed += 1
        env.test_num = completed
        progress["stage"] = "environment_close"
        env.close_env(clear_cache=completed % args["clear_cache_freq"] == 0)
        if env.render_freq:
            env.viewer.close()
        print(f"{job['task']} {job['task_config']}: completed {completed}/{job['test_num']}, "
              f"seed {seed}, success {result['success']}", flush=True)
    model.client.close()


def summarize(run_dir: Path, identity: dict) -> dict:
    settings = {}
    for setting in identity["task_configs"]:
        tasks = []
        for task in identity["tasks"]:
            directory = run_dir / setting / task
            episodes = records(directory / "episodes")
            attempts = records(directory / "attempts")
            errors = records(directory / "errors")
            successes = sum(item["success"] for item in episodes)
            tasks.append({
                "task": task, "completed": len(episodes), "successes": successes,
                "target": identity["test_num"], "missing": max(0, identity["test_num"] - len(episodes)),
                "success_rate": successes / len(episodes) if episodes else None,
                "completed_seeds": [item["seed"] for item in episodes],
                "skipped_candidates": sum(item["status"] == "skipped" for item in attempts),
                "environment_errors": [item for item in errors if item["category"] == "environment"],
                "model_errors": [item for item in errors if item["category"] == "model"],
                "worker_exits": records(directory / "worker_runs"),
                "videos": [item["video"] for item in episodes if item["video"]],
            })
        completed = sum(item["completed"] for item in tasks)
        successes = sum(item["successes"] for item in tasks)
        settings[setting] = {"completed": completed, "successes": successes,
                             "target": len(tasks) * identity["test_num"],
                             "success_rate": successes / completed if completed else None,
                             "missing_tasks": [item["task"] for item in tasks if item["missing"]], "tasks": tasks}
    result = {"run_id": identity["run_id"], "settings": settings}
    write_json(run_dir / "summary.json", result)
    return result


def stop_server(process) -> None:
    if process.poll() is None:
        process.terminate()
        process.wait()


class ChildProcesses:
    """Own the model and simulator processes for one controller invocation."""

    def __init__(self):
        self.processes = []
        self.previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in self.previous_handlers:
            signal.signal(sig, self.handle_signal)
        atexit.register(self.terminate)

    def terminate(self):
        for process in reversed(self.processes):
            stop_server(process)

    def handle_signal(self, signum, _frame):
        self.terminate()
        sys.exit(128 + signum)

    def close(self):
        self.terminate()
        for sig, handler in self.previous_handlers.items():
            signal.signal(sig, handler)
        atexit.unregister(self.terminate)


def run_controller(args) -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    root = Path(args.robotwin_root).resolve()
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if args.task_list:
        tasks = [line.strip() for line in Path(args.task_list).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    elif args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        tasks = manifest["selected_tasks"] if "selected_tasks" in manifest else manifest["tasks"]
    else:
        # Read the actual installed RoboTwin task catalog using its own YAML dependency.
        catalog = subprocess.check_output([
            args.robotwin_py, "-c", "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))['tasks']))",
            str(root / "env_cfg" / "eval" / "all_tasks.yml"),
        ], text=True)
        tasks = json.loads(catalog)
    tasks = list(dict.fromkeys(tasks))
    train_config = json.loads(Path(args.run_config).read_text())
    architecture = {name: train_config[name] for name in (
        "training_mode", "action_horizon", "model_action_dim", "max_token_len", "normalize_method",
        "paligemma_variant", "action_expert_variant")}
    start_seed = args.start_seed if args.start_seed is not None else 100000 * (1 + args.seed)
    identity = {
        "run_id": args.run_id, "tasks": tasks, "task_configs": args.task_configs,
        "test_num": args.test_num, "start_seed": start_seed, "instruction_type": args.instruction_type,
        "deterministic_instruction": args.deterministic_instruction, "exec_steps": args.exec_steps,
        "params_path": str(Path(args.params_path).resolve()), "run_config": train_config,
        "move_norm_stats_path": str(Path(args.move_norm_stats_path).resolve()),
        "operate_norm_stats_path": str(Path(args.operate_norm_stats_path).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()), "architecture": architecture,
        "num_steps": args.num_steps, "robotwin_root": str(root), "robotwin_revision": revision, "video": args.video,
    }
    run_dir = Path(args.output_root).resolve() / args.run_id
    run_file = run_dir / "run.json"
    if run_file.exists() and json.loads(run_file.read_text()) != identity:
        write_json(run_dir / "requested_run.json", identity)
        print(f"RUN_ID has different settings; use a new run ID. Requested settings: {run_dir / 'requested_run.json'}")
        return
    write_json(run_file, identity)
    summary = summarize(run_dir, identity)
    if all(not item["missing_tasks"] for item in summary["settings"].values()):
        print(json.dumps(summary, indent=2))
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.model_gpus
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    service_id = uuid.uuid4().hex
    command = [sys.executable, "-m", "mto.infer", "--params-path", identity["params_path"],
               "--move-norm-stats-path", identity["move_norm_stats_path"],
               "--operate-norm-stats-path", identity["operate_norm_stats_path"],
               "--tokenizer-path", identity["tokenizer_path"], "--host", args.server_host,
               "--port", str(args.server_port), "--num-steps", str(args.num_steps), "--service-id", service_id]
    for name, value in architecture.items():
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    children = ChildProcesses()
    with (run_dir / "server.log").open("a") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.processes.append(server)
    deadline = time.monotonic() + args.startup_timeout
    ready = False
    while server.poll() is None and time.monotonic() < deadline:
        response = subprocess.run(["curl", "--silent", "--fail", "--max-time", "2",
                                   f"http://{args.server_host}:{args.server_port}/healthz"], capture_output=True)
        if response.returncode == 0 and response.stdout.strip() == b"OK":
            ready = True
            break
        time.sleep(2)
    if not ready:
        write_json(run_dir / "service_status.json", {"status": "not_ready", "log": str(run_dir / "server.log")})
        children.close()
        print(f"Service did not become ready; results remain incomplete. See {run_dir / 'server.log'}")
        return
    client = WebsocketClientPolicy(args.server_host, args.server_port)
    metadata = client.get_server_metadata()
    client.close()
    matches = (
        metadata["action_dim"] == 14 and metadata["action_type"] == "absolute_qpos"
        and metadata["action_representation"] == "joint14_delta_joint_absolute_gripper"
        and metadata["state_layout"] == "left_arm6,left_gripper,right_arm6,right_gripper"
        and metadata["normalization_mode"] == "per-expert"
        and metadata["normalization_phases"] == ["move", "operate"]
        and metadata["service_id"] == service_id
        and metadata["action_horizon"] == architecture["action_horizon"]
        and 1 <= args.exec_steps <= metadata["action_horizon"]
        and identity["move_norm_stats_path"] != identity["operate_norm_stats_path"]
    )
    write_json(run_dir / "service_status.json", {"status": "ready" if matches else "contract_mismatch",
                                                "metadata": metadata, "service_id": service_id, "command": command})
    if not matches:
        children.close()
        print(f"Service contract does not match this evaluation. See {run_dir / 'service_status.json'}")
        return

    env["CUDA_VISIBLE_DEVICES"] = args.sim_gpus
    env["PYTHONPATH"] = os.pathsep.join([str(Path(args.mto_root).resolve() / "src"), env.get("PYTHONPATH", "")])
    seed_stop = start_seed + (args.max_seed_attempts or args.test_num * 50)
    for setting in args.task_configs:
        for task in tasks:
            directory = run_dir / setting / task
            directory.mkdir(parents=True, exist_ok=True)
            while len(records(directory / "episodes")) < args.test_num:
                settled = [item["seed"] for item in records(directory / "attempts")
                           if item["status"] in {"completed", "skipped", "candidate_error"}]
                settled.extend(item["seed"] for item in records(directory / "episodes"))
                next_seed = max(settled, default=start_seed - 1) + 1
                if next_seed >= seed_stop:
                    break
                job = {**identity, "task": task, "task_config": setting, "directory": str(directory),
                       "next_seed": next_seed, "seed_stop": seed_stop, "mto_root": str(Path(args.mto_root).resolve()),
                       "server_host": args.server_host, "server_port": args.server_port}
                write_json(directory / "job.json", job)
                with (directory / "worker.log").open("a") as log:
                    worker = subprocess.Popen([args.robotwin_py, str(Path(__file__).resolve()),
                                               "--worker-job", str(directory / "job.json")],
                                              env=env, stdout=log, stderr=subprocess.STDOUT)
                    children.processes.append(worker)
                    worker.wait()
                write_json(directory / "worker_runs" / f"{time.time_ns()}.json", {
                    "start_seed": next_seed, "exit_code": worker.returncode,
                    "completed_after": len(records(directory / "episodes")), "log": str(directory / "worker.log"),
                })
                summarize(run_dir, identity)
                attempts = records(directory / "attempts")
                last = max(attempts, key=lambda item: item["seed"], default={})
                # Exit codes are diagnostics only. Restart after a scene-selection
                # exception; a policy/rollout error leaves this task incomplete.
                if last.get("status") != "candidate_error" or last["seed"] < next_seed:
                    break
                print(f"{setting}/{task}: scene seed {last['seed']} skipped after {last['stage']}; "
                      f"worker exit {worker.returncode}", flush=True)
    children.close()
    summary = summarize(run_dir, identity)
    for setting, result in summary["settings"].items():
        print(f"{setting}: {result['successes']} successes / {result['completed']} completed / "
              f"{result['target']} requested; missing tasks: {result['missing_tasks']}")
    print(f"Full results: {run_dir / 'summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--robotwin-py", required=True)
    parser.add_argument("--mto-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--move-norm-stats-path", required=True)
    parser.add_argument("--operate-norm-stats-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="eval_results")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-list", help="One task name per line; default: RoboTwin's all_tasks.yml catalog")
    selection.add_argument("--manifest", help="MTO dataset manifest selected_tasks, or explicit JSON tasks list")
    parser.add_argument("--task-configs", nargs="+", default=["demo_clean"])
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--instruction-type", choices=["seen", "unseen"], default="unseen")
    parser.add_argument("--deterministic-instruction", action="store_true")
    parser.add_argument("--exec-steps", type=int, default=30)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow decoding steps")
    parser.add_argument("--max-seed-attempts", type=int)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-gpus", default="0")
    parser.add_argument("--sim-gpus", default="0")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9700)
    parser.add_argument("--startup-timeout", type=float, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-job":
        run_worker(Path(sys.argv[2]))
    else:
        run_controller(parse_args())
