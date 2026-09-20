# LeRobot training and RoboTwin evaluation

This is an implementation handoff, not a verified reproduction of the paper's training recipe. The README deliberately remains minimal. All commands below use local datasets and assets after environment installation; substitute the indicated deployment paths.

## Environment and local assets

Keep the MTO JAX environment separate from the RoboTwin simulation environment. From the MTO checkout on Linux, create the locked Python 3.11 environment:

```bash
cd /mnt/pfs/public/xuhaoming/Move-then-Operate
uv sync --extra cuda12 --frozen
```

The reader uses PyArrow and PyAV (including AV1 decoding), without a LeRobot package dependency. Torch supplies CPU data loading. RoboTwin needs its own installed simulator, robot/task assets, and lightweight client packages `numpy`, `msgpack`, `websockets`, and `typing_extensions`. Its environment does not need JAX.

Set local paths once:

```bash
export MTO_DIR=/mnt/pfs/public/xuhaoming/Move-then-Operate
export MTO_PY="$MTO_DIR/.venv/bin/python"
export DATA_ROOT=/mnt/pfs/public/fanyupeng/dataset/robotwin2_lerobot
export LABELS_ROOT=/mnt/pfs/public/xuhaoming/mto_artifacts/labels
export RUN_ROOT=/mnt/pfs/public/xuhaoming/mto_artifacts
export PI0_PARAMS=/mnt/pfs/public/xuhaoming/mto_artifacts/base_assets/openpi-assets/checkpoints/pi0_base/params
export TOKENIZER=/root/.cache/openpi/big_vision/paligemma_tokenizer.model
cd "$MTO_DIR"
```

`PI0_PARAMS` must be an actual JAX parameter directory. If unavailable, the separate download command is `"$MTO_PY" -m mto.download_assets --output-dir "$RUN_ROOT/base_assets"`; use the paths it prints. Tokenizer and base weights remain independent assets; large base weights are not copied into run metadata.

## Label all tasks

Omit `--task_names` to include every task (the intended collection has 50). The first run uses **all 50 tasks, clean training only**. Labeling, statistics, and training all explicitly select `demo_clean`. Keep randomized evaluation separate from the training split.

```bash
"$MTO_PY" -m mto.auto_label \
  --data_format lerobot_v3 --root_dir "$DATA_ROOT" \
  --labels_root "$LABELS_ROOT" \
  --split_names demo_clean \
  --model_id ep-20260605100618-g5rhc \
  --camera_key observation.images.cam_high \
  --sample_fps 5 --max_frames 64 --max_new_tokens 8192 \
  --concurrency 5 --max_attempts 3
```

The API key is supplied through `ARK_API_KEY`. Annotation uses the OpenAI SDK's `client.responses.create` against `https://ark.cn-beijing.volces.com/api/v3`, with endpoint `ep-20260605100618-g5rhc` by default. Frame indices use `input_text`, and frames use JPEG data URLs in `input_image`; only the response's final `output_text` becomes label JSON. This follows the [Ark Responses interface](https://www.volcengine.com/docs/82379/1795150). `--max_new_tokens` maps to `max_output_tokens` and includes thinking plus final output; its default is 8192. The locked environment already includes a compatible SDK, so use `uv sync --extra cuda12 --frozen` after updating the checkout instead of upgrading the SDK independently.

Existing valid annotations with matching sidecars are resumed. The sidecar records the API type, model endpoint and token limit; labels from a different request configuration are archived and regenerated. If your annotations are already complete, proceed directly to statistics; the training reader records missing/invalid labels.

## Compute two expert statistics

```bash
export NORM_DIR="$RUN_ROOT/norm/all_tasks_clean_h30"
"$MTO_PY" -m mto.compute_norm_stats \
  --data-format lerobot_v3 --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" --split-names demo_clean \
  --action-horizon 30 --normalize-method zscore \
  --num-samples-per-expert 50000 --seed 42 --output-dir "$NORM_DIR"
```

This produces **different** `move.json` and `operate.json` statistics plus `data_manifest.json`. Inspect the selected datasets, phase counts and missing/invalid annotations in the manifest. The default includes no small-action exclusion. Recompute statistics if you change the training split, labels, horizon or action filtering.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$MTO_PY" -m mto.train \
  --data-format lerobot_v3 --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" --split-names demo_clean \
  --exp-name mto_all_tasks_clean_h30_lora --checkpoint-base-dir "$RUN_ROOT/checkpoints" \
  --init-params "$PI0_PARAMS" --init-source pi0_base --tokenizer-path "$TOKENIZER" \
  --training-mode lora --action-horizon 30 --model-action-dim 32 \
  --batch-size 8 --num-workers 4 --num-steps 10000 \
  --lr-warmup 1000 --lr 5e-5 --lr-final 1e-5 --ema-decay 0.99 \
  --long-phase-ratio 0.5 --normalize-method zscore \
  --move-norm-stats-path "$NORM_DIR/move.json" \
  --operate-norm-stats-path "$NORM_DIR/operate.json" \
  --save-interval 1000 --keep-period 5000 --fsdp-devices 1 --seed 42
```

Batch 8 is a starting configuration; VRAM feasibility has not been measured on the remote A100. Training modes are `full`, `frozen_vlm`, `lora`, and `lora_frozen_vlm`. The example chooses LoRA because that was used in the later experiments. Change the experiment name when changing model structure or training mode. Global batch must divide across visible JAX devices. Single-process, multi-GPU training is supported by the existing mesh; there is no multi-host launch implementation or gradient accumulation option.

`pi0_base` loads the pretrained VLM and copies the action Transformer into both experts, initializing each expert's action/state/time projections and the router independently. `dual_expert` loads an existing dual-expert model. Omitting `--init-params` starts a random model. Resume restores model, optimizer, EMA and step; it does not restore the exact data iterator or all random states. Python, NumPy, Torch and workers are seeded, but interruption/resume is not bitwise equivalent to uninterrupted training.

The run directory contains:

```text
checkpoints/wide_camera_dual_expert/mto_all_tasks_clean_h30_lora/
  resolved_config.json
  data_manifest.json
  assets/move.json
  assets/operate.json
  10000/params/
  10000/train_state/
```

The manifests record actual episode/phase inclusion and omissions. Statistics are archived on new runs and reused from that run on resume. Resume invocation/configuration is recorded separately. Resume reads the saved model, data selection, normalization, seed and optimizer settings; current CLI values cannot silently replace that contract. Only operational settings (workers, logging/saving, device mesh, total step budget) remain configurable. Use a new experiment to change the training contract. With EMA enabled, exported `params` are EMA parameters. LoRA checkpoints currently contain the full model tree, not only adapter deltas. Pass the `params` directory to inference.

## Serve a trained checkpoint

```bash
export TRAIN_RUN="$RUN_ROOT/checkpoints/wide_camera_dual_expert/mto_all_tasks_clean_h30_lora"
export PARAMS="$TRAIN_RUN/10000/params"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$MTO_PY" -m mto.infer \
  --params-path "$PARAMS" --tokenizer-path "$TOKENIZER" \
  --training-mode lora --action-horizon 30 --model-action-dim 32 \
  --move-norm-stats-path "$TRAIN_RUN/assets/move.json" \
  --operate-norm-stats-path "$TRAIN_RUN/assets/operate.json" \
  --normalize-method zscore --num-steps 10 --host 127.0.0.1 --port 9700
```

Use the saved architecture, including any explicit variant overrides. The service compiles both expert decoders before exposing `/healthz`; compilation time is printed separately. Each real request selects exactly one action expert and keeps it fixed throughout that chunk. One service is intended for one active environment. `reset(seed)` resets its model noise at each accepted episode seed.

The adapter sends three current RGB images, the raw 14D joint drive-target state, and the environment's task instruction. It executes returned absolute qpos commands with RoboTwin `take_action(..., action_type="qpos")`, observes again after `exec_steps`, and stops on success or the environment action limit. `exec_steps` must be no greater than the checkpoint horizon. No ground-truth phase or future state is sent to the model.

## MTO-owned RoboTwin evaluation

`mto.eval_robotwin` implements the evaluation loop itself; it does not call RoboTwin's legacy or XPolicyLab evaluation runner. The environment API was aligned to [RoboTwin main at 6dde571](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/scripts/eval_policy_xpolicylab.py). It uses the installed environment's `envs.CONFIGS_PATH` for task, embodiment and camera configuration (including both `task_config/` and `env_cfg/task_config/` layouts), screens scene seeds with the environment expert, then resets the same accepted scene for the MTO policy. The official environment still determines task success and action budgets.

Stop any manually started service from the previous example before using this wrapper, which manages its own service:

```bash
export ROBOTWIN_ROOT=/root/xuhaoming/xr-2/RoboTwin
export ROBOTWIN_PY=/root/xuhaoming/xr-2/.venv/bin/python
export PARAMS_PATH="$TRAIN_RUN/10000/params"
export RUN_CONFIG="$TRAIN_RUN/resolved_config.json"
export MOVE_NORM_STATS_PATH="$TRAIN_RUN/assets/move.json"
export OPERATE_NORM_STATS_PATH="$TRAIN_RUN/assets/operate.json"
export TOKENIZER_PATH="$TOKENIZER"
export TASK_MANIFEST="$TRAIN_RUN/data_manifest.json"
export EVAL_ROOT="$RUN_ROOT/eval"

RUN_ID=mto_clean10k_eval_clean TEST_NUM=100 EXEC_STEPS=30 \
TASK_CONFIGS=demo_clean MODEL_GPUS=0 SIM_GPUS=0 SERVER_PORT_BASE=9700 \
bash "$MTO_DIR/eval_robotwin_mto_resumable.sh"

RUN_ID=mto_clean10k_eval_random TEST_NUM=100 EXEC_STEPS=30 \
TASK_CONFIGS=demo_randomized MODEL_GPUS=0 SIM_GPUS=0 SERVER_PORT_BASE=9800 \
bash "$MTO_DIR/eval_robotwin_mto_resumable.sh"
```

`TASK_MANIFEST` uses the selected training task list. Use `TASK_MANIFEST` on 8600, where the newer `env_cfg/eval/all_tasks.yml` catalog is absent. `TASK_LIST` can instead name a local text file with one task per line, or `TASK_CATALOG` a YAML file containing a `tasks` list (CLI: `--task-catalog`). With no explicit selection the newer default catalog is used if present; otherwise the command prints the available selection options. `TASK_CONFIGS` remains a setting name such as `demo_clean`, not a filesystem path. A single GPU is used sequentially by default. Simulator and model GPU visibility can be configured separately.

Repeat the same command and `RUN_ID` to resume. Each task records candidate seeds, accepted completed episodes, environment/model errors, action/route traces and videos. Only success or action-budget termination counts as a completed trial. Scene-selection errors skip that candidate as in the reference environment workflow; an error during policy rollout leaves the trial missing and stops that task until a later retry. A bounded candidate-seed budget prevents endlessly searching an unsatisfiable task. Summary SR is successes divided by completed trials, with requested totals and missing tasks reported separately. Clean and Random summaries remain separate. A partial result is never described as a complete 50×100 evaluation.

Results live under `EVAL_ROOT/RUN_ID`: `run.json`, `service_status.json`, `server.log`, `summary.json`, and `<setting>/<task>/{episodes,attempts,errors,traces,videos,worker_runs}`. Reusing a run ID with different checkpoint/configuration settings is rejected with the requested settings saved for inspection. Source/configuration identity is recorded without hashing checkpoint contents.

The model is reset using the actual accepted environment seed, rather than the trial counter. `--start-seed` selects the candidate range. Stable instruction selection is available via `--deterministic-instruction`; use the same instruction policy and seed range for comparisons. Without this option, instruction generation follows the environment's random selection and may differ across process restarts.

## Validation scope

Local integration tests cover synthetic Parquet/video data, CPU checkpoint save/restore/continue with and without EMA, and a stub policy/environment for the transport/runner path. Run the bounded checks with:

```bash
cd "$MTO_DIR"
"$MTO_PY" -m unittest discover -s tests
```

These local checks do not establish full-model GPU restoration or SAPIEN execution. On 8600, continue the existing validation checkpoint from step 2 to step 3, load the new checkpoint in a fresh inference process, and complete a single-task RoboTwin episode before scaling to all tasks. Keep engineering test labels separate from the semantic annotations used for formal training. Offline router accuracy and action loss are training diagnostics, not closed-loop success rates.
