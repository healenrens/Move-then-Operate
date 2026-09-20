# Data preparation and annotation

MTO trains on synchronized robot episodes stored in HDF5. The preparation utility exports annotation videos from existing episodes; robot demonstration collection is performed outside this repository.

Run the commands below in the project environment described in the [README](../README.md). Replace the example absolute data paths and task instructions with your own.

## Directory layout

```text
/data/mto_data/
└── move_can_pot/
    └── demo_clean/
        ├── data/
        │   ├── episode0.hdf5
        │   └── episode1.hdf5
        ├── instructions/
        │   ├── episode0.json
        │   └── episode1.json
        ├── video/
        │   ├── episode0.mp4
        │   └── episode1.mp4
        └── auto_labels_v2/
            ├── episode0_phases_labels_thinking.json
            └── episode1_phases_labels_thinking.json
```

Use the same `episodeN` basename for the HDF5 file, instruction file, video, and annotation, where `N` is an integer. The training loader searches recursively for `data/` directories beneath its data root. Preparation and annotation each operate on one episode group, such as `/data/mto_data/move_can_pot/demo_clean`.

An instruction file contains task-level language:

```json
{
  "seen": [
    "Move the can into the pot."
  ]
}
```

If `seen` contains several equivalent instructions, annotation and training select from that list independently. The training prompt comes from this file, not from the generated subtask descriptions.

## HDF5 schema

Each episode contains `T` synchronized steps:

| HDF5 dataset | Shape | Meaning |
| --- | --- | --- |
| `/joint_action/left_arm` | `[T, 6]` | Absolute left-arm joint positions in radians |
| `/joint_action/right_arm` | `[T, 6]` | Absolute right-arm joint positions in radians |
| `/joint_action/left_gripper` | `[T]` or `[T, 1]` | Absolute left-gripper opening in `[0, 1]` |
| `/joint_action/right_gripper` | `[T]` or `[T, 1]` | Absolute right-gripper opening in `[0, 1]` |
| `/observation/head_camera/rgb` | `[T]` encoded buffers | One JPEG image per step |
| `/observation/left_camera/rgb` | `[T]` encoded buffers | One JPEG image per step |
| `/observation/right_camera/rgb` | `[T]` encoded buffers | One JPEG image per step |

Image entries are JPEG byte buffers, such as HDF5 variable-length `uint8` arrays. They are not decoded `[H, W, 3]` arrays. All three cameras and all joint arrays use the same frame index. There is no required front camera or end-effector pose dataset.

The loader forms the 14-dimensional state as:

```text
[left joints (6), left gripper (1), right joints (6), right gripper (1)]
```

It reads both the current state and future joint targets from `joint_action`. Supply absolute joint positions in these datasets, rather than velocities or precomputed deltas.

## Export annotation videos

Preparation writes every selected camera frame once, in its original order. Video frame `i` therefore corresponds to HDF5 step `i`; no frame sampling occurs during export.

The frame rate must be supplied explicitly. This example assumes episodes recorded at 30 FPS; use the actual recording rate of your data:

```bash
python -m mto.prepare_data \
  --root-dir /data/mto_data/move_can_pot/demo_clean \
  --fps 30 \
  --camera head_camera \
  --instruction "Move the can into the pot."
```

`--camera` defaults to `head_camera`; `left_camera` and `right_camera` are also supported. Existing videos are skipped unless `--overwrite` is supplied. `--instruction` creates missing `instructions/episodeN.json` files and preserves existing files. Omit it when instruction files are already prepared or need episode-specific wording.

Existing frame-aligned videos can be used directly. Their frames must retain the HDF5 indices: trimming, reordering, or changing the number of frames changes the label-to-action correspondence.

## Generate phase annotations

Set `ARK_API_KEY` in the environment for the Doubao service, then run:

```bash
python -m mto.auto_label \
  --root_dir /data/mto_data/move_can_pot/demo_clean \
  --model_id doubao-seed-1-6-thinking-250715 \
  --sample_fps 5 \
  --max_frames 64 \
  --max_new_tokens 2048 \
  --concurrency 5 \
  --max_attempts 3
```

The annotator uses the video's recorded FPS and includes original frame indices in the model input. `--sample_fps` and `--max_frames` control the existing frame sampler, which also includes endpoint frames. The default output directory is `auto_labels_v2` beside `data/`, `video/`, and `instructions/`. Keep this location for direct training-loader compatibility.

Existing nonempty annotation JSON files are skipped by default. Use `--no_resume` to regenerate them. `--videos_limit` selects a prefix of the sorted video list. Raw model responses are written beside the output annotations.

The JSON format uses inclusive start and end frame indices. For a 120-frame example with one movement followed by one operation, a complete annotation is:

```json
[
  {
    "subtask": 1,
    "subtask_description": "Move the can into the pot.",
    "primary_arm": "right",
    "start_frame_idx": 0,
    "end_frame_idx": 119,
    "target_object_name": "can",
    "target_object_axis": [0.42, 0.53],
    "left_gripper_end_axis": [-1.0, -1.0],
    "right_gripper_end_axis": [0.31, 0.58],
    "phases": [
      {
        "phase_type": "move",
        "phase_description": "Move the right gripper toward the can.",
        "start_frame_idx": 0,
        "end_frame_idx": 59
      },
      {
        "phase_type": "operate",
        "phase_description": "Grasp the can and place it into the pot.",
        "start_frame_idx": 60,
        "end_frame_idx": 119
      }
    ]
  }
]
```

Subtasks and their phases should cover the trajectory consecutively. `phase_type` is `move` or `operate`; `primary_arm` is `left`, `right`, `both`, or `unknown`. Image coordinates are normalized with the top-right corner at `(0, 0)` and the bottom-left at `(1, 1)`. An absent gripper uses `[-1, -1]`. Coordinates and descriptions are annotation metadata; training uses the phase boundaries and the task-level instruction.

The loader converts inclusive label ends to exclusive slice ends once when constructing its phase pools. A phase needs at least two recorded states to provide one action target. Shorter phases are omitted, and sampled action chunks remain within their phase. `move` maps to route label `0`; `operate` maps to route label `1`.

## Action representation

For a chunk beginning at step `s`, arm action targets are `q[s+k] - q[s]`, for future steps `k = 1, ..., H`. Both gripper dimensions retain their future absolute values. Each chunk uses one phase label. Short chunks are padded and accompanied by action masks.

At inference, reverse normalization using the selected expert's statistics, then add the chunk's original starting joint positions to each arm target. Do not cumulatively sum the predicted chunk. Gripper values are neither delta-converted nor normalized by the dataset.

## Compute normalization assets

Compute statistics from the labeled training split. The utility uses the same dataset phase records and statistics functions as training and does not read cached statistics from the working directory.

For separate movement and operation statistics:

```bash
python -m mto.compute_norm_stats \
  --data-root /data/mto_data \
  --task-name move_can_pot \
  --output-dir /data/mto_assets/move_can_pot \
  --mode per-expert \
  --normalize-method zscore \
  --action-horizon 30
```

This writes `move.json` and `operate.json`. The selected data must contain both phase types. Each file contains 14-dimensional action and proprioception statistics. Internal phase names `long` and `short` correspond to `move` and `operate`.

For shared statistics:

```bash
python -m mto.compute_norm_stats \
  --data-root /data/mto_data \
  --output-dir /data/mto_assets/shared \
  --mode shared \
  --normalize-method zscore \
  --action-horizon 30
```

This writes `shared.json`. Omitting `--task-name` includes all tasks beneath the data root. Repeating a command overwrites the files in its output directory.

The preserved statistics collector computes arm deltas relative to each labeled phase's starting state, while training chunks use their own starting state. It records mean, standard deviation, minimum, maximum, and the 1st/99th percentiles. `zscore` is the training default; `min_max` uses the recorded percentiles. Gripper dimensions 6 and 13 remain in their original scale in either mode.

Pass `move.json` and `operate.json` through the training options `move_norm_stats_path` and `operate_norm_stats_path` with shared normalization disabled. For shared normalization, pass `shared.json` through `shared_norm_stats_path`. Use the same normalization method in training and inference. Explicit paths load the requested files directly; leaving paths unset uses the dataset's existing working-directory statistics loading or calculation behavior.
