"""One integration test: real WebSocket transport, stub policy, fake RoboTwin.

Run: .venv/bin/python -m unittest discover -s tests -p test_eval_pipeline.py
No JAX model, checkpoint, SAPIEN, rendering or robot assets are loaded.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from mto import eval_robotwin


_SERVER = '''
import json
from pathlib import Path
import sys
import numpy as np
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

events = Path(sys.argv[3])
def record(item):
    with events.open("a") as stream:
        stream.write(json.dumps(item) + "\\n")

class StubPolicy:
    def reset(self, seed=None):
        self.seed, self.chunk = seed, 0
        record({"event": "reset", "seed": seed})

    def infer(self, obs):
        assert set(obs) == {"images", "state", "prompt"}
        assert obs["state"].shape == (14,)
        for name, color in (("head_camera", 17), ("left_camera", 33), ("right_camera", 65)):
            image = obs["images"][name]
            assert image.shape == (4, 5, 3) and image.dtype == np.uint8
            assert np.all(image == color)
        record({"event": "infer", "seed": self.seed, "chunk": self.chunk,
                "state": obs["state"].tolist(), "prompt": obs["prompt"]})
        actions = np.full((4, 14), 100 + self.seed + self.chunk, dtype=np.float32)
        actions += np.arange(4, dtype=np.float32)[:, None] / 10
        actions[:, 6], actions[:, 13] = -0.25, 1.25
        route = self.chunk % 2
        self.chunk += 1
        return {"actions": actions, "actions_delta": np.full((4, 14), -999, np.float32),
                "phase": ("move", "operate")[route], "route_probabilities": np.eye(2)[route]}

metadata = {"action_horizon": 4, "action_dim": 14, "action_type": "absolute_qpos",
            "action_representation": "joint14_delta_joint_absolute_gripper",
            "state_layout": "left_arm6,left_gripper,right_arm6,right_gripper",
            "normalization_mode": "per-expert", "normalization_phases": ["move", "operate"],
            "service_id": sys.argv[2]}
WebsocketPolicyServer(StubPolicy(), host="127.0.0.1", port=int(sys.argv[1]), metadata=metadata).serve_forever()
'''

_ENVIRONMENT = '''
import json
import os
from pathlib import Path
import numpy as np

events = Path(__file__).resolve().parents[1] / "environment_events.jsonl"
def record(item):
    with events.open("a") as stream:
        stream.write(json.dumps(item) + "\\n")

class fake_task:
    def setup_demo(self, *, now_ep_num, seed, is_test, **args):
        record({"event": "setup", "seed": seed})
        # Exercise a reference scene error followed by a native simulator crash.
        assert seed != 20, "fixture expert scene error"
        if seed == 21:
            os._exit(9)
        self.seed = seed
        self.take_action_cnt, self.step_lim = 0, 3
        self.eval_success = False
        self.plan_success = seed != 10
        self.render_freq = 0
        self.state = np.arange(14, dtype=np.float32) / 20

    def play_once(self):
        return {"info": {"seed": self.seed}}

    def check_success(self):
        return self.plan_success

    def close_env(self, **kwargs):
        pass

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        return {"observation": {name: {"rgb": np.full((4, 5, 3), color, np.uint8)}
                for name, color in (("head_camera", 17), ("left_camera", 33), ("right_camera", 65))},
                "joint_action": {"vector": self.state.copy()}}

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        self.state = action.copy()
        self.take_action_cnt += 1
        self.eval_success = self.seed == 11
        record({"event": "action", "seed": self.seed, "step": self.take_action_cnt,
                "action": action.tolist()})
'''


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class EvalPipelineIntegrationTest(unittest.TestCase):
    def test_websocket_rollout_results_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="mto-eval-pipeline-") as temporary:
            root = Path(temporary).resolve()
            robotwin = root / "RoboTwin"
            configs = robotwin / "env_cfg/task_config"
            _write(robotwin / "envs/__init__.py", f"CONFIGS_PATH = {str(configs) + '/'!r}\n")
            _write(robotwin / "envs/fake_task.py", _ENVIRONMENT)
            _write(robotwin / "description/utils/generate_episode_instructions.py", '''
def generate_episode_descriptions(task, episodes, count):
    return [{"seen": [f"seen {episodes[0]['seed']}"], "unseen": [f"unseen {episodes[0]['seed']}"]}]
''')
            _write(configs / "demo_clean.yml", {"data_type": {}, "camera": {"head_camera_type": "fixture"},
                                                "embodiment": ["fixture"], "render_freq": 0, "clear_cache_freq": 5})
            _write(configs / "_camera_config.yml", {"fixture": {"h": 4, "w": 5}})
            _write(configs / "_embodiment_config.yml", {"fixture": {"file_path": str(robotwin / "robot")}})
            _write(robotwin / "robot/config.yml", {})
            _write(robotwin / "env_cfg/eval/all_tasks.yml", {"tasks": ["fake_task"]})
            _write(root / "resolved_config.json", {
                "training_mode": "full", "action_horizon": 4, "model_action_dim": 32, "max_token_len": 50,
                "normalize_method": "zscore", "paligemma_variant": None, "action_expert_variant": None,
            })
            _write(root / "stub_server.py", _SERVER)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            repository = Path(__file__).resolve().parents[1]
            args = SimpleNamespace(
                robotwin_root=str(robotwin), robotwin_py=sys.executable, mto_root=str(repository),
                params_path=str(root / "unused_params"), run_config=str(root / "resolved_config.json"),
                move_norm_stats_path=str(root / "move.json"), operate_norm_stats_path=str(root / "operate.json"),
                tokenizer_path=str(root / "unused_tokenizer"), run_id="resume", output_root=str(root / "results"),
                task_list=None, manifest=None, task_configs=["demo_clean"], test_num=2, seed=0, start_seed=10,
                instruction_type="unseen", deterministic_instruction=True, exec_steps=2, num_steps=10,
                max_seed_attempts=2, video=False, model_gpus="", sim_gpus="", server_host="127.0.0.1",
                server_port=port, startup_timeout=20,
            )
            real_popen, real_run = subprocess.Popen, subprocess.run
            launched_models, launched_workers = [], []

            def popen(command, *positional, **kwargs):
                if command[1:3] == ["-m", "mto.infer"]:
                    launched_models.append(command)
                    command = [sys.executable, str(root / "stub_server.py"),
                               command[command.index("--port") + 1], command[command.index("--service-id") + 1],
                               str(root / "server_events.jsonl")]
                elif "--worker-job" in command:
                    launched_workers.append(json.loads(Path(command[-1]).read_text())["next_seed"])
                return real_popen(command, *positional, **kwargs)

            def run(command, *positional, **kwargs):
                if command[:3] == ["git", "-C", str(robotwin)]:
                    return subprocess.CompletedProcess(command, 0, "fake-environment-revision\n", "")
                return real_run(command, *positional, **kwargs)

            run_dir = root / "results/resume"
            directory = run_dir / "demo_clean/fake_task"
            with mock.patch.object(subprocess, "Popen", side_effect=popen), \
                    mock.patch.object(subprocess, "run", side_effect=run), \
                    mock.patch.dict(os.environ, {"PYTHONPATH": str(repository / "src")}), \
                    contextlib.redirect_stdout(io.StringIO()):
                eval_robotwin.run_controller(args)
                first = json.loads((run_dir / "summary.json").read_text())["settings"]["demo_clean"]
                self.assertEqual((first["completed"], first["successes"], first["target"]), (1, 1, 2),
                                 (directory / "worker.log").read_text())
                self.assertEqual(first["missing_tasks"], ["fake_task"])
                self.assertEqual(eval_robotwin.records(directory / "worker_runs")[0]["exit_code"], 0)
                self.assertEqual(eval_robotwin.records(directory / "attempts")[0]["status"], "skipped")
                original_episode = (directory / "episodes/seed_11.json").read_bytes()
                args.max_seed_attempts = 3
                eval_robotwin.run_controller(args)
                self.assertEqual((directory / "episodes/seed_11.json").read_bytes(), original_episode)
                self.assertEqual(launched_workers, [10, 12])
                final = json.loads((run_dir / "summary.json").read_text())["settings"]["demo_clean"]
                self.assertEqual((final["completed"], final["successes"], final["success_rate"]), (2, 1, 0.5))
                self.assertEqual(final["missing_tasks"], [])
                # Fully completed resume does not even launch another model server.
                eval_robotwin.run_controller(args)
                self.assertEqual(len(launched_models), 2)

                # A fresh worker dying natively must not reuse an older candidate_error
                # as a reason to retry the same seed forever.
                args.run_id, args.start_seed, args.test_num, args.max_seed_attempts = "native_crash", 20, 1, 3
                eval_robotwin.run_controller(args)
                self.assertEqual(launched_workers[-2:], [20, 21])
                crash_dir = root / "results/native_crash/demo_clean/fake_task"
                self.assertEqual(eval_robotwin.records(crash_dir / "episodes"), [])
                self.assertEqual([item["exit_code"] for item in eval_robotwin.records(crash_dir / "worker_runs")], [1, 9])

            episodes = {item["seed"]: item for item in eval_robotwin.records(directory / "episodes")}
            self.assertEqual((episodes[11]["termination"], episodes[11]["steps"]), ("success", 1))
            self.assertEqual((episodes[12]["termination"], episodes[12]["steps"]), ("step_limit", 3))
            self.assertEqual(episodes[12]["route_counts"], {"move": 1, "operate": 1})
            self.assertEqual(episodes[12]["route_switches"], 1)
            self.assertEqual([episodes[seed]["gripper_clipped_values"] for seed in (11, 12)], [2, 6])
            trace = _jsonl(directory / "traces/seed_12.jsonl")
            actions = [item for item in trace if item["event"] == "action"]
            self.assertEqual([item["chunk"] for item in actions], [0, 0, 1])
            np.testing.assert_allclose([item["action"][0] for item in actions], [112, 112.1, 113])
            self.assertEqual([item["action"][6] for item in actions], [0, 0, 0])
            self.assertEqual([item["action"][13] for item in actions], [1, 1, 1])
            self.assertEqual(trace[-1]["event"], "terminal")
            server_events = _jsonl(root / "server_events.jsonl")
            self.assertEqual([item["seed"] for item in server_events if item["event"] == "reset"], [11, 12])
            requests = [item for item in server_events if item["event"] == "infer"]
            self.assertEqual([(item["seed"], item["chunk"]) for item in requests], [(11, 0), (12, 0), (12, 1)])
            np.testing.assert_allclose(requests[1]["state"], np.arange(14, dtype=np.float32) / 20)
            self.assertEqual(requests[1]["prompt"], "unseen 12")
            status = json.loads((run_dir / "service_status.json").read_text())
            self.assertEqual(status["status"], "ready")
            self.assertEqual(status["service_id"], status["metadata"]["service_id"])


if __name__ == "__main__":
    unittest.main()
