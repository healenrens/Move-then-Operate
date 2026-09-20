# Episode data and phase labels

MTO reads LeRobot **v3.0 on-disk data directly with PyArrow and PyAV**. It does not use the installed LeRobot Python package. Legacy raw HDF5 episodes remain available with `--data-format hdf5`. Training and statistics share `PhaseDataset` and `build_action_window` in `src/mto/dataset.py`.

## LeRobot layout

```text
DATA_ROOT/task/demo_clean/                  # or demo_randomized
  meta/info.json
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  videos/observation.images.cam_high/chunk-*/file-*.mp4
  videos/observation.images.cam_left_wrist/chunk-*/file-*.mp4
  videos/observation.images.cam_right_wrist/chunk-*/file-*.mp4
```

Every metadata shard is enumerated. `info.json` supplies the data/video path templates; episode metadata supplies their chunk/file indices. Rows are selected with `episode_index` and the global `[dataset_from_index,dataset_to_index)` interval, then sorted by episode-local `frame_index`. A global index is never used as a shard-local row number. `tasks.parquet` stores the task instruction in its pandas string index; its Arrow metadata identifies that column.

The reader expects `observation.state` and `action` as float32 joint14 arrays, plus `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index`. The state layout is `[left_arm6, left_gripper, right_arm6, right_gripper]`. A shared video may contain multiple episodes. Frame `i` is decoded at the episode's `videos/<camera>/from_timestamp + rows[i].timestamp`, using actual presentation timestamps. All cameras are RGB, resized directly to 224×224 with PIL bicubic interpolation and scaled to [-1,1].

Task and split selection is exact. An empty task list includes every task; an empty split list includes every split found. Selecting a single `task/split` directory retains the same identity as selecting it under the whole collection.

## Labels

Local labels live at `DATA_ROOT/task/split/auto_labels_v2/episodeN_phases_labels_thinking.json`. An external `LABELS_ROOT` mirrors `task/split/auto_labels_v2/...`. Thus `episode0` from different tasks cannot collide. Annotation writes a matching `.meta.json` sidecar containing source identity, length, FPS, camera, instruction, sampling settings, and frame convention.

Labels are arrays of subtasks with integer `start_frame_idx`, `end_frame_idx`, and a `phases` array. Each phase has `phase_type` equal to `move` or `operate` and the same two bounds. Bounds are **inclusive episode-local state frames**. Subtasks must cover the complete episode consecutively, and phases must cover their subtask consecutively, without gaps or overlaps. A pure-operate subtask is valid. A one-state phase is valid annotation but contributes no action chunks. The loader reports missing/invalid labels and single-state phases in `data_manifest.json`; inspect these counts before considering the selected collection fully labeled.

Auto-labeling samples uniformly over the **whole episode**, includes the endpoints, and respects both target FPS and the maximum frame count. Resume requires valid label content and matching sidecar metadata, not just a nonempty file. A missing or incompatible sidecar causes re-annotation. Training accepts structurally valid existing labels without requiring a sidecar. API/JSON failures get bounded retries; invalid coverage is never repaired with invented phases. Task instructions are the training language inputs; generated descriptions and future observations are not model inputs.

## Action windows and normalization

For a phase spanning states `[a,b]`, sample anchor `a <= s < b` and set `n = min(H,b-s)`:

- LeRobot absolute targets: `action[s:s+n]`. These already correspond to the next states; do not shift them again.
- HDF5 absolute targets: `state[s+1:s+n+1]`.
- Subtract `state[s]` from target dimensions `[0,1,2,3,4,5,7,8,9,10,11,12]`.
- Leave grippers 6 and 13 absolute. Zero-valued absolute targets are valid data.
- Pad to `H` and mask padded rows out of the loss. Only the 14 physical dimensions contribute; padding to model dimension 32 is masked too.

Move is route 0; operate is route 1. With both present, the default sampler chooses each with probability 0.5, then chooses a phase uniformly within that expert and an anchor uniformly in its valid interval. Phases can yield short chunks near their ends. The boundary-crossing transition is excluded. There is no implicit action filtering or time compression. Optional `--drop-small-action-deltas` excludes near-stationary rows from loss/statistics while preserving their time slots. If every row in a training batch is excluded, flow loss is zero and router supervision remains. A selected expert must still have valid action rows for statistics estimation.

**Each expert has its own statistics file**, `move.json` and `operate.json`. The statistics tool samples the exact same raw window function, conditioned on the expert, with a fixed seed. It collects only valid action rows and one raw state per chunk; padding and dimensions 14–31 never enter statistics. The sample count is configurable, so these are empirical estimates of the training distribution. Metadata records the selected datasets, horizon, sampling, seed, action representation and filtering. There is no implicit working-directory cache or automatic statistics computation during training.

Training normalizes each sample with its labeled expert's file. Inference chooses the expert from the visual/language context, normalizes the current state with that expert's file, runs only that expert, and reverses normalization with the same file. It then adds the original chunk-start joints to every predicted delta. Do not cumulatively sum the chunk. Both normalization modes (`zscore`, percentile `min_max`) leave grippers unchanged. The RoboTwin adapter clips commanded grippers to [0,1] and records any clipping.

## Legacy HDF5

```text
DATA_ROOT/task/split/data/episodeN.hdf5
DATA_ROOT/task/split/instructions/episodeN.json
DATA_ROOT/task/split/video/episodeN.mp4
```

HDF5 contains `/joint_action/{left_arm,right_arm}` shaped `[T,6]`, `/joint_action/{left_gripper,right_gripper}` shaped `[T]` or `[T,1]`, and `/observation/{head_camera,left_camera,right_camera}/rgb` with one encoded JPEG buffer per frame. All arrays use the same clock. Instructions use `{"seen": ["Task instruction."]}`; training selects the first instruction deterministically.

Export videos without changing frame indices:

```bash
python -m mto.prepare_data --root-dir /data/mto/task/demo_clean --fps 30 --camera head_camera
```

Use the real recording FPS. Annotate these videos with `mto.auto_label --data_format hdf5 --root_dir /data/mto/task/demo_clean`. Statistics and training then use `--data-format hdf5`, with the same two normalization files and phase-window contract.

See [the run guide](lerobot_robotwin.md) for environment setup and commands.
