# Move-then-Operate

JAX implementation of a hard-switch, two-expert vision-language-action policy built on [OpenPI](https://github.com/Physical-Intelligence/openpi). A shared vision-language backbone predicts **move** or **operate**. During training, phase labels select each sample's expert loss. During inference, the policy selects one expert per action chunk and reuses the shared prefix KV cache throughout flow integration.

This repository contains the phase annotator, the 14-dimensional joint-position data pipeline, training configurations, and single-robot inference. It does not include the original trained checkpoints or demonstrations. The release has been reorganized from the research implementation; GPU retraining and checkpoint-based inference still need to be evaluated on the training server.

## Repository layout

```text
src/mto/                 Annotation, data preparation, statistics, training, inference
src/openpi/models/       JAX dual-expert model, Gemma, SigLIP, LoRA, tokenizer
src/openpi/training/     Optimizer, sharding, checkpoints, weight loading
src/openpi/policies/     Hard-switch MTO inference policy
src/openpi_client/       WebSocket client and array serialization
docs/data.md             HDF5 schema, annotation format, data preparation
pyproject.toml           Environment and package definitions
uv.lock                  Resolved dependencies
```

## 1. Set up the environment

The GPU training target is **Linux x86-64 with NVIDIA GPUs**, using Python 3.11 and the CUDA 12 JAX wheels. Use an NVIDIA driver compatible with those wheels. Core versions are JAX 0.5.3, Flax 0.10.2, Optax 0.2.4, and Orbax 0.11.13. PyTorch/torchvision are used for CPU data loading and image transforms; the model and optimizer run in JAX. There is no LeRobot, Transformers, or PyTorch model dependency.

```bash
git clone https://github.com/healenrens/Move-then-Operate.git
cd Move-then-Operate
python3 -m pip install --user uv
uv sync --frozen --extra cuda12
source .venv/bin/activate
```

`uv` installs Python 3.11 if it is not already available. The lockfile selects CPU PyTorch wheels on Linux, independently of the JAX CUDA backend. For annotation or data preparation on a CPU machine, use `uv sync --frozen` without `--extra cuda12`.

After installation, these commands describe the available options without starting training:

```bash
python -m mto.train --help
python -m mto.auto_label --help
python -m mto.infer --help
```

## 2. Prepare and annotate demonstrations

The input is an existing collection of robot HDF5 episodes. Robot data collection and simulator demonstration generation are outside this repository. See [the data guide](docs/data.md) for the complete schema and an annotation example.

```text
/data/mto_dataset/pick_can/demo_clean/
├── data/episode0.hdf5
├── instructions/episode0.json
├── video/episode0.mp4
└── auto_labels_v2/episode0_phases_labels_thinking.json
```

Each instruction JSON uses `{"seen": ["Pick up the can."]}`. The HDF5 episode contains left/right six-joint positions, two gripper values, and three JPEG camera streams named `head_camera`, `left_camera`, and `right_camera`.

Generate annotation videos from the HDF5 camera stream. Set `--fps` to the actual recording cadence; the following example assumes 30 FPS. Every video frame retains its original HDF5 index.

```bash
python -m mto.prepare_data \
  --root-dir /data/mto_dataset/pick_can/demo_clean \
  --fps 30 \
  --instruction "Pick up the can."
```

Existing instruction files are preserved. Omit `--instruction` when per-episode instruction files already exist. This command processes one demonstration directory; repeat it for other tasks/directories.

The annotator uses an OpenAI-compatible vision API. Supply your API key and an available vision model ID from your provider. The original model ID remains the default, but availability depends on your account.

```bash
read -r -s -p "Annotation API key: " ARK_API_KEY
export ARK_API_KEY
python -m mto.auto_label \
  --root_dir /data/mto_dataset/pick_can/demo_clean \
  --base_url https://ark.cn-beijing.volces.com/api/v3 \
  --model_id doubao-seed-1-6-thinking-250715 \
  --sample_fps 5 \
  --max_frames 64 \
  --concurrency 5
```

Keep the default `auto_labels_v2/` output location so the trainer can discover labels. `root_dir` is a single directory containing `video/` and `instructions/`; annotation is not recursive. Existing nonempty label files are skipped; pass `--no_resume` to regenerate them. Frame indices in the JSON are inclusive, zero-based indices into the original episode. Review the phase boundaries before training.

## 3. Compute normalization statistics

Compute statistics after annotation, reusing the dataset's phase statistics collector:

```bash
python -m mto.compute_norm_stats \
  --data-root /data/mto_dataset \
  --output-dir /data/mto_assets/norm_stats \
  --mode shared
```

The shared mode writes `shared.json`. Use `--mode per-expert` to write `move.json` and `operate.json`. These files are external assets and are not embedded in checkpoints.

The 14-dimensional ordering is `[left joints (6), left gripper, right joints (6), right gripper]`. Joint action targets are deltas from the **initial state of the chunk**, not cumulative deltas between successive predictions. Grippers remain absolute. Joint state/action dimensions are normalized; gripper dimensions are not. The examples use z-score normalization.

## 4. Obtain initialization assets

To start a new training run from the public pi0 base, download its parameters and tokenizer once:

```bash
python -m mto.download_assets --output-dir /data/mto_assets
```

This produces the paths used below:

```text
/data/mto_assets/openpi-assets/checkpoints/pi0_base/params/
/data/mto_assets/big_vision/paligemma_tokenizer.model
```

Alternatively, provide existing local files at any path. Training and inference commands use local asset paths; they do not need to download model weights during a run.

## 5. Start training

The four presets control which backbone parameters are trained. Expert projections and the router are trainable in all presets.

| `--training-mode` | Vision-language backbone | Action Transformers |
| --- | --- | --- |
| `full` | All parameters | All parameters |
| `frozen_vlm` | Frozen | All parameters |
| `lora` | Language LoRA; image encoder frozen | LoRA |
| `lora_frozen_vlm` | Frozen | LoRA |

Example: a fresh run using a frozen VLM and LoRA action experts. Adjust batch size to the available GPU memory; no fixed VRAM requirement is claimed for these presets.

```bash
CUDA_VISIBLE_DEVICES=0 python -m mto.train \
  --data-root /data/mto_dataset \
  --exp-name mto_lora_run1 \
  --checkpoint-base-dir /data/mto_checkpoints \
  --training-mode lora_frozen_vlm \
  --init-source pi0_base \
  --init-params /data/mto_assets/openpi-assets/checkpoints/pi0_base/params \
  --tokenizer-path /data/mto_assets/big_vision/paligemma_tokenizer.model \
  --shared-norm-stats-path /data/mto_assets/norm_stats/shared.json \
  --normalize-method zscore \
  --action-horizon 30 \
  --batch-size 8 \
  --fsdp-devices 1 \
  --num-workers 8 \
  --num-steps 30000 \
  --save-interval 1000 \
  --ema-decay None \
  --no-resume
```

`data-root` is searched recursively for `**/data/episode*.hdf5` and the matching labels. Default sampling selects move/operate with equal probability when both exist. The physical action dimension is 14; model inputs/outputs are padded to 32, with padding excluded from the action loss. Time padding is also excluded. Router cross-entropy is averaged over samples.

For separate move/operate statistics, replace `--shared-norm-stats-path` with:

```text
--no-use-shared-action-norm-stats
--move-norm-stats-path /data/mto_assets/norm_stats/move.json
--operate-norm-stats-path /data/mto_assets/norm_stats/operate.json
```

For multi-GPU training on one machine, expose the desired GPUs and set `--fsdp-devices` to their count. The global batch size must be divisible by the number of exposed devices. For example, use `CUDA_VISIBLE_DEVICES=0,1,2,3`, `--fsdp-devices 4`, and `--batch-size 16` with the same command. Multi-host launch orchestration is not included.

W&B and the expensive per-expert gradient diagnostics are disabled by default. Enable W&B with `--wandb-enabled --project-name YOUR_PROJECT` after configuring W&B on the server. `--log-gradient-diagnostics` enables additional forward/backward passes.

### Initialization versus resume

- **New run from pi0:** `--init-source pi0_base`. Copies the VLM and the action Transformer into both experts; independently initializes each expert's action/state/time projections and the router.
- **New run from an MTO checkpoint:** `--init-source dual_expert --init-params /path/to/STEP/params`, with a new experiment name. Restores the two experts and their projections; newly requested LoRA adapters can be initialized. Optimizer state starts fresh.
- **Continue the same run:** repeat its command with `--resume` instead of `--no-resume`. Keep experiment name, checkpoint directory, architecture, normalization, and optimizer settings consistent. Resume restores parameters, optimizer, step, and EMA directly; it does not use `init-params` or partially initialize missing parameters. `num-steps` is the total target step count.

Checkpoints are written under:

```text
/data/mto_checkpoints/wide_camera_dual_expert/mto_lora_run1/STEP/
├── params/        # Model parameters for inference (EMA parameters when enabled)
└── train_state/   # Training state for resume
```

Keep the matching normalization JSON, tokenizer, and training configuration alongside your run. LoRA checkpoints currently contain the full model parameter tree, not an adapter-only export. `--ema-decay None` avoids a second EMA parameter copy during training.

## 6. Run hard-switch inference

Use the same architecture, horizon, tokenizer, and normalization files as the training run. The following command starts a WebSocket server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mto.infer \
  --params-path /data/mto_checkpoints/wide_camera_dual_expert/mto_lora_run1/30000/params \
  --training-mode lora_frozen_vlm \
  --action-horizon 30 \
  --tokenizer-path /data/mto_assets/big_vision/paligemma_tokenizer.model \
  --move-norm-stats-path /data/mto_assets/norm_stats/shared.json \
  --operate-norm-stats-path /data/mto_assets/norm_stats/shared.json \
  --normalize-method zscore \
  --host 0.0.0.0 \
  --port 8000
```

Replace `30000` with an actual saved step. For separate statistics, pass `move.json` and `operate.json` respectively. To process one local observation instead, append `--observation-path /data/observation.npz --output-path /data/actions.npz` to the same command.

NPZ input keys are `head_camera`, `left_camera`, and `right_camera` (HWC uint8 RGB), `state` (14 raw qpos values), and `prompt` (a scalar string). WebSocket observations use:

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
result = policy.infer({
    "images": {
        "head_camera": head_rgb,
        "left_camera": left_rgb,
        "right_camera": right_rgb,
    },
    "state": current_qpos,
    "prompt": "Pick up the can.",
})
actions = result["actions"]  # [horizon, 14], absolute joint/gripper commands
```

Outputs also include `phase`, `route_index`, `route_probabilities`, and `actions_delta` (joint deltas with absolute grippers). The robot client decides how many actions to execute before acquiring a new observation. The selected expert stays fixed throughout each generated chunk. Both experts' weights remain loaded, but only the selected expert performs action computation. No Aloha coordinate conversion is applied.

## Attribution

The JAX model components, training utilities, and WebSocket interfaces are adapted from Physical Intelligence's OpenPI. Upstream license and copyright notices are retained. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
