"""Phase-local action chunks shared by training and per-expert statistics.

Labels use inclusive state-frame bounds [a,b]. Anchors are uniform in [a,b),
then n=min(H,b-anchor). LeRobot action[anchor:anchor+n] already points to the
next states; HDF5 uses state[anchor+1:anchor+n+1]. No target crosses a phase.
"""
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from mto.labels import label_path, load_phase_labels
from mto.normalization import ACTION_REPRESENTATION, JOINT_INDICES, QposNormalizer

CAMERAS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


@dataclass(frozen=True)
class Hdf5Episode:
    dataset_root: Path
    relative_id: str
    episode_index: int
    length: int
    path: Path


class Hdf5Reader:
    def __init__(self, data_root, task_names=(), split_names=()):
        root = Path(data_root).resolve()
        self.episodes = []
        for path in sorted(root.glob("**/data/episode*.hdf5")):
            dataset_root = path.parent.parent
            relative_id = f"{dataset_root.parent.name}/{dataset_root.name}"
            if task_names and dataset_root.parent.name not in task_names:
                continue
            if split_names and dataset_root.name not in split_names:
                continue
            with h5py.File(path, "r") as hf:
                length = len(hf["joint_action/left_arm"])
            self.episodes.append(Hdf5Episode(dataset_root, relative_id, int(path.stem[7:]), length, path))

    def get_instruction(self, episode):
        path = episode.dataset_root / "instructions" / f"episode{episode.episode_index}.json"
        return json.loads(path.read_text())["seen"][0]

    def read_rows(self, episode, start=0, stop=None):
        with h5py.File(episode.path, "r") as hf:
            arrays = [np.asarray(hf[f"joint_action/{key}"][start:stop]).reshape(-1, size)
                      for key, size in (("left_arm", 6), ("left_gripper", 1), ("right_arm", 6), ("right_gripper", 1))]
        return {"observation.state": np.concatenate(arrays, axis=-1).astype(np.float32)}

    def read_rgb(self, episode, camera_key, local_frame_indices):
        camera = {v: k for k, v in CAMERAS.items()}[camera_key]
        with h5py.File(episode.path, "r") as hf:
            return [Image.open(io.BytesIO(hf[f"observation/{camera}/rgb"][i])).convert("RGB")
                    for i in local_frame_indices]


def build_action_window(state, absolute_actions, horizon, *, drop_small_action_deltas=False, min_action_magnitude=1e-5):
    """Convert one chunk relative to its anchor, retaining zero-valued targets."""
    actions = np.asarray(absolute_actions, np.float32).copy()
    actions[:, JOINT_INDICES] -= state[JOINT_INDICES]
    valid = np.ones(len(actions), dtype=bool)
    if drop_small_action_deltas:
        # Optional loss exclusion preserves temporal slots; it never compacts time.
        motion = np.asarray(absolute_actions) - state
        valid = np.max(np.abs(motion), axis=-1) >= min_action_magnitude
    padded = np.zeros((horizon, 14), np.float32)
    mask = np.zeros(horizon, dtype=bool)
    padded[:len(actions)] = actions
    mask[:len(actions)] = valid
    return padded, mask


def image_to_tensor(image):
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image, np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0


class PhaseDataset(Dataset):
    def __init__(self, data_dir, action_steps=30, image_transform=None, normalize_actions=True,
                 data_format="lerobot_v3", task_names=(), split_names=(), labels_root=None,
                 normalize_method="zscore", long_phase_ratio=0.5,
                 drop_small_action_deltas=False, min_action_magnitude=1e-5,
                 move_norm_stats_path=None, operate_norm_stats_path=None):
        if data_format == "auto":
            data_format = "lerobot_v3" if next(Path(data_dir).glob("**/meta/info.json"), None) else "hdf5"
        if data_format in ("lerobot", "lerobot_v3"):
            from mto.lerobot_reader import LeRobotReader
            self.reader = LeRobotReader(data_dir, task_names, split_names)
            self.data_format = "lerobot_v3"
        else:
            self.reader = Hdf5Reader(data_dir, task_names, split_names)
            self.data_format = "hdf5"
        self.action_steps = action_steps
        self.image_transform = image_transform or image_to_tensor
        self.long_phase_ratio = long_phase_ratio
        self.drop_small_action_deltas = drop_small_action_deltas
        self.min_action_magnitude = min_action_magnitude
        self.pools = {"move": [], "operate": []}
        self.manifest = {
            "data_root": str(Path(data_dir).resolve()), "data_format": self.data_format,
            "task_names": list(task_names), "split_names": list(split_names),
            "labels_root": str(Path(labels_root).resolve()) if labels_root else None,
            "action_representation": ACTION_REPRESENTATION, "action_horizon": action_steps,
            "sampling": "phase uniform within expert; anchor uniform in [phase_start,phase_end)",
            "move_probability": long_phase_ratio, "drop_small_action_deltas": drop_small_action_deltas,
            "min_action_magnitude": min_action_magnitude,
            "episodes": [], "missing_labels": [], "invalid_labels": [], "single_frame_phases": 0,
        }
        for episode in self.reader.episodes:
            path = label_path(episode.dataset_root, episode.relative_id, episode.episode_index,
                              Path(labels_root) if labels_root else None)
            identity = {"dataset": episode.relative_id, "episode_index": episode.episode_index,
                        "length": episode.length, "label_path": str(path)}
            if not path.exists():
                self.manifest["missing_labels"].append(identity)
                continue
            labels, error = load_phase_labels(path, episode.length)
            if error:
                self.manifest["invalid_labels"].append({**identity, "error": error})
                continue
            phase_list = []
            for subtask in labels:
                for phase in subtask["phases"]:
                    entry = {"start": phase["start_frame_idx"], "end": phase["end_frame_idx"],
                             "phase_type": phase["phase_type"].lower(), "episode": episode}
                    phase_list.append({k: v for k, v in entry.items() if k != "episode"})
                    if entry["start"] == entry["end"]:
                        self.manifest["single_frame_phases"] += 1
                    else:
                        self.pools[entry["phase_type"]].append(entry)
            self.manifest["episodes"].append({**identity, "phases": phase_list})
        self.manifest["phase_counts"] = {key: len(pool) for key, pool in self.pools.items()}
        self.manifest["selected_tasks"] = sorted({ep.relative_id.split("/")[0] for ep in self.reader.episodes})
        self.manifest["selected_datasets"] = sorted({ep.relative_id for ep in self.reader.episodes})
        self.normalizers = None
        if normalize_actions:
            self.normalizers = {"move": QposNormalizer(move_norm_stats_path, normalize_method),
                                "operate": QposNormalizer(operate_norm_stats_path, normalize_method)}
        print(f"Dataset: {len(self.manifest['episodes'])}/{len(self.reader.episodes)} labeled episodes; "
              f"phases={self.manifest['phase_counts']}, missing={len(self.manifest['missing_labels'])}, "
              f"invalid={len(self.manifest['invalid_labels'])}, "
              f"single-frame={self.manifest['single_frame_phases']}")

    def __len__(self):
        return sum(len(pool) for pool in self.pools.values())

    def sample_raw(self, rng, phase_type=None, *, images=False):
        if phase_type is None:
            if self.pools["move"] and self.pools["operate"]:
                phase_type = "move" if rng.random() < self.long_phase_ratio else "operate"
            else:
                phase_type = "move" if self.pools["move"] else "operate"
        entry = rng.choice(self.pools[phase_type])
        anchor = rng.randrange(entry["start"], entry["end"])
        return self.raw_window(entry, anchor, images=images)

    def raw_window(self, entry, anchor, *, images=False):
        episode = entry["episode"]
        n = min(self.action_steps, entry["end"] - anchor)
        rows = self.reader.read_rows(episode, anchor, anchor + n + 1)
        state = np.asarray(rows["observation.state"][0], np.float32)
        absolute = rows["action"][:n] if self.data_format == "lerobot_v3" else rows["observation.state"][1:n+1]
        action, mask = build_action_window(state, absolute, self.action_steps,
                                          drop_small_action_deltas=self.drop_small_action_deltas,
                                          min_action_magnitude=self.min_action_magnitude)
        sample = {"state": state, "action": action, "mask": mask, "phase_type": entry["phase_type"],
                  "metadata": {"dataset": episode.relative_id, "episode_index": episode.episode_index,
                               "phase_start": entry["start"], "phase_end": entry["end"],
                               "window_start": anchor, "valid_targets": n}}
        if images:
            sample["images"] = {name: self.reader.read_rgb(episode, key, [anchor])[0]
                                for name, key in CAMERAS.items()}
            sample["instruction"] = self.reader.get_instruction(episode)
        return sample

    def __getitem__(self, index):
        raw = self.sample_raw(random, images=True)
        phase = raw["phase_type"]
        state, action, mask = raw["state"], raw["action"], raw["mask"]
        if self.normalizers is not None:
            state = self.normalizers[phase].normalize_state(state)
            action = self.normalizers[phase].normalize(action, "action")
        action[~mask] = 0
        return {"observation": {"state": torch.from_numpy(state), "instr": raw["instruction"],
                                "image": {k: self.image_transform(v) for k, v in raw["images"].items()}},
                "action": torch.from_numpy(action), "action_mask": torch.from_numpy(mask),
                "action_loss_mask": torch.from_numpy(mask.copy()), "action_dim_mask": torch.ones(14, dtype=torch.bool),
                "route_label": torch.tensor(0 if phase == "move" else 1, dtype=torch.int64),
                "phase_type": phase, "metadata": raw["metadata"]}


# Kept for callers of the original HDF5 training entry point.
class WideCameraDataset(PhaseDataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(data_dir, data_format="hdf5", **kwargs)


def wide_camera_collate_fn(batch):
    result = {key: torch.stack([item[key] for item in batch]) for key in
              ("action", "action_mask", "action_loss_mask", "action_dim_mask", "route_label")}
    result["observation"] = {
        "state": torch.stack([item["observation"]["state"] for item in batch]),
        "instr": [item["observation"]["instr"] for item in batch],
        "image": {key: torch.stack([item["observation"]["image"][key] for item in batch]) for key in CAMERAS},
    }
    result["metadata"] = [item["metadata"] for item in batch]
    result["phase_type"] = [item["phase_type"] for item in batch]
    return result
