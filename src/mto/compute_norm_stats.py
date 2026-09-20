"""Sample the training chunk distribution separately for move and operate."""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from mto.dataset import PhaseDataset
from mto.normalization import ACTION_REPRESENTATION, JOINT_INDICES, detailed_statistics


def compute_statistics(dataset, phase, num_samples, seed):
    rng = random.Random(seed)
    actions, states = [], []
    for _ in range(num_samples):
        raw = dataset.sample_raw(rng, phase)
        actions.append(raw["action"][raw["mask"]])
        states.append(raw["state"])
    actions = np.concatenate(actions)
    return detailed_statistics(actions, np.stack(states)), len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-format", choices=["auto", "hdf5", "lerobot_v3", "lerobot"], default="lerobot_v3")
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-names", nargs="*", default=[])
    parser.add_argument("--split-names", nargs="*", default=[])
    parser.add_argument("--normalize-method", choices=["zscore", "min_max"], default="zscore")
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--num-samples-per-expert", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-small-action-deltas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-action-magnitude", type=float, default=1e-5)
    args = parser.parse_args()
    dataset = PhaseDataset(args.data_root, data_format=args.data_format, labels_root=args.labels_root,
                           task_names=args.task_names, split_names=args.split_names,
                           action_steps=args.action_horizon, normalize_actions=False,
                           drop_small_action_deltas=args.drop_small_action_deltas,
                           min_action_magnitude=args.min_action_magnitude)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_manifest.json").write_text(json.dumps(dataset.manifest, indent=2) + "\n")
    for phase in ("move", "operate"):
        statistics, action_rows = compute_statistics(dataset, phase, args.num_samples_per_expert, args.seed)
        payload = {
            "normalization_config": {
                "version": "mto_chunk_v2", "normalization_mode": "per-expert", "phase": phase,
                "method": args.normalize_method, "action_representation": ACTION_REPRESENTATION,
                "action_reference": "chunk_anchor", "joint_delta_indices": JOINT_INDICES.tolist(),
                "absolute_gripper_indices": [6, 13], "action_horizon": args.action_horizon,
                "sampling": dataset.manifest["sampling"], "seed": args.seed,
                "num_samples": args.num_samples_per_expert, "valid_action_rows": action_rows,
                "state_rows": args.num_samples_per_expert, "padding_included": False,
                "data_format": dataset.data_format, "data_root": str(args.data_root.resolve()),
                "selected_datasets": dataset.manifest["selected_datasets"],
                "drop_small_action_deltas": args.drop_small_action_deltas,
                "min_action_magnitude": args.min_action_magnitude,
            }, "statistics": statistics,
        }
        output = args.output_dir / f"{phase}.json"
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {output}: {args.num_samples_per_expert} chunks, {action_rows} valid action rows")


if __name__ == "__main__":
    main()
