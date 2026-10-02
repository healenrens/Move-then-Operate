# LeRobot and RoboTwin deployment contract

Use the [README](../README.md) for installation, annotation, statistics, full/LoRA training, resume, inference and evaluation commands. Use [data.md](data.md) for the exact action alignment and phase boundaries.

MTO reads LeRobot v3.0 through PyArrow/PyAV, independently of the LeRobot training package. Mount complete datasets, including episode metadata, Parquet shards and all three camera videos. External labels mirror task/split identity. Use the same task/split selection for annotation, statistics and training; compute statistics from training episodes only.

Keep the MTO Python 3.11/JAX environment separate from the installed RoboTwin simulator environment. Across containers, each process must see its own configured paths; the simulator needs the inference server's reachable host/port. Mount checkpoints, tokenizer and the two archived expert statistics in the model container. Mount RoboTwin code, simulator assets and results in the simulator container. Run manifests and logs are deployment artifacts, not publication files.

The evaluation wrapper starts and manages its inference service. It reads the training architecture from `resolved_config.json`, uses the run's `assets/move.json` and `assets/operate.json`, and checks the active service identity before dispatching workers. Full and LoRA checkpoints need matching model structures. Each action chunk uses one inferred route throughout flow integration.

The MTO-owned controller screens candidate scenes with the environment expert, resets the accepted scene for the policy, and records its actual seed. It sends current RGB/joint14/instruction observations and executes absolute qpos commands. Scene-selection errors skip candidates; policy-rollout errors leave that trial incomplete for a later retry. Only success or action-budget termination counts as completed. Native simulator crashes remain visible in worker output. Candidate selection is bounded. Summary success rate divides successes by completed trials and separately reports requested/missing trials.

Task selection can come from a training manifest, a local task-list text file, or a YAML task catalog. `demo_clean`/`demo_randomized` are settings, not guarantees of independent seeds. Use separate run IDs for different settings or checkpoints. An episode holdout needs separate, metadata-complete exports; there is no built-in random episode splitter or validation loader.

Offline integration checks use controlled CPU models and a simulator stub. Remote verification should establish full-model checkpoint continuation, fresh-process inference with the matching architecture/statistics, and actual SAPIEN rollout before scaling to all tasks. Record the deployed RoboTwin revision, checkpoint, task selection, settings and completed-trial counts with results.
