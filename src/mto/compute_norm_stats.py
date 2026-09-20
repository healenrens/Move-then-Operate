"""Compute MTO normalization assets from labeled training episodes."""

import argparse
import json
from pathlib import Path

from mto.dataset import ActionNormalizer, WideCameraDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="", help="Optional task path filter.")
    parser.add_argument("--mode", choices=["per-expert", "shared"], default="per-expert")
    parser.add_argument("--normalize-method", choices=["zscore", "min_max"], default="zscore")
    parser.add_argument("--action-horizon", type=int, default=30)
    args = parser.parse_args()

    dataset = WideCameraDataset(
        data_dir=str(args.data_root),
        action_steps=args.action_horizon,
        normalize_actions=False,
        data_type="qpos",
        task_name=[args.task_name] if args.task_name else [],
        use_shared_action_norm_stats=args.mode == "shared",
    )
    phases = [("global", "shared.json")] if args.mode == "shared" else [
        ("long", "move.json"),
        ("short", "operate.json"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for phase, filename in phases:
        actions, proprios = dataset._collect_all_actions_and_proprios(phase=phase)
        normalizer = ActionNormalizer("qpos_delta", phase)
        statistics = normalizer._compute_detailed_stats(actions, proprios)
        payload = {
            "normalization_config": {
                "method": args.normalize_method,
                "data_type": "qpos_delta",
                "phase": phase,
                "action_reference": "phase_start",
                "version": "mto_qpos_v1",
            },
            "statistics": statistics,
        }
        output_path = args.output_dir / filename
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote normalization statistics: {output_path}")


if __name__ == "__main__":
    main()
