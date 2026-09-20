import torch
import h5py
import numpy as np
import glob
import json
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import io
from typing import Dict, List, Any, Tuple

# ==================================================================================================
# Helper Functions
# ==================================================================================================

DELTA_EPS = 1e-6
SMALL_ACTION_EPS = 1e-5
_QPOS_DELTA_MASK = np.array([True, True, True, True, True, True, False,
                             True, True, True, True, True, True, False], dtype=bool)


def _ensure_2d(array: np.ndarray, width: int) -> np.ndarray:
    if array.ndim == 2:
        return array
    return array.reshape(-1, width)



def biased_randint(start: int, end: int, min_tail_ratio: float = 0.5, bias_towards: str = "low") -> int:
    """
    在闭区间 [start, end] 上进行非均匀整数采样：
    - 越靠近 `bias_towards` 一端，采样概率越高（线性下降）。
    - 另一端的最小概率不低于最高概率的 `min_tail_ratio`。

    实现方式：构造首项为 1、末项为 `min_tail_ratio` 的等差权重，
    使用前缀和 + 二分查找做逆变换采样。

    Args:
        start: 区间起点（含）。
        end: 区间终点（含）。
        min_tail_ratio: 尾端最小概率占头端最大概率的比例，(0, 1]。
        bias_towards: 'low' 表示靠近 `start` 概率更高；'high' 表示靠近 `end` 概率更高。

    Returns:
        采样得到的整数。
    """
    if start >= end:
        return int(start)
    if not (0 < min_tail_ratio <= 1):
        min_tail_ratio = max(1e-6, min(1.0, float(min_tail_ratio)))

    lo, hi = int(start), int(end)
    n = hi - lo + 1

    if bias_towards == 'low':
        a0, an_1 = 1.0, float(min_tail_ratio)
    else:
        a0, an_1 = float(min_tail_ratio), 1.0
    d = (an_1 - a0) / (n - 1)

    total = n * (a0 + an_1) / 2.0

    def prefix(k: int) -> float:
        wk = a0 + k * d
        return (k + 1) * (a0 + wk) / 2.0

    import random as _random
    u = _random.random() * total
    left, right = 0, n - 1
    while left < right:
        mid = (left + right) // 2
        if prefix(mid) >= u:
            right = mid
        else:
            left = mid + 1
    return lo + left






def _apply_qpos_delta_np(actions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Convert absolute qpos actions to deltas relative to base state (grippers stay absolute)."""
    if actions.size == 0:
        return actions
    delta_actions = actions.astype(np.float32, copy=True)
    if base is None or base.size == 0:
        return delta_actions
    delta_actions[:, _QPOS_DELTA_MASK] -= base[_QPOS_DELTA_MASK]
    zero_rows = np.max(np.abs(actions), axis=1) < DELTA_EPS
    
    # 对于全0的动作帧，使用上一帧的delta值（保持动作增量不变）
    for i in range(len(delta_actions)):
        if zero_rows[i]:
            if i > 0:
                # 复制上一帧的delta值
                delta_actions[i] = delta_actions[i - 1].copy()
            else:
                # 第一帧没有上一帧，保持为0
                delta_actions[i] = 0
    return delta_actions


def _apply_qpos_delta_tensor(actions: torch.Tensor, mask: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Convert padded tensor qpos actions to deltas relative to base state (grippers absolute)."""
    if actions.numel() == 0:
        return actions
    if mask is None or mask.numel() == 0:
        return actions

    valid_idx = mask.nonzero(as_tuple=False).flatten()
    delta_actions = actions.clone()
    if valid_idx.numel() == 0:
        delta_actions.zero_()
        return delta_actions

    base = base.to(actions.device, dtype=actions.dtype)
    delta_mask = torch.as_tensor(_QPOS_DELTA_MASK, dtype=torch.bool, device=actions.device)

    valid_actions = delta_actions[valid_idx]
    valid_actions[:, delta_mask] -= base[delta_mask]

    original_valid = actions[valid_idx]
    zero_rows = original_valid.abs().max(dim=1).values < DELTA_EPS
    
    # 对于全0的动作帧，使用上一帧的delta值（保持动作增量不变）
    for i in range(len(valid_actions)):
        if zero_rows[i]:
            if i > 0:
                # 复制上一帧的delta值
                valid_actions[i] = valid_actions[i - 1].clone()
            else:
                # 第一帧没有上一帧，保持为0
                valid_actions[i] = 0

    delta_actions[valid_idx] = valid_actions
    delta_actions[~mask] = 0
    return delta_actions

# ==================================================================================================
# ActionNormalizer Class
# ==================================================================================================

class ActionNormalizer:
    """动作和状态的归一化处理器"""

    def __init__(self, data_type, phase: str = ""):
        self.stats = {}
        self.normalized = False
        self.data_type = data_type
        self.phase = phase or "global"

    def compute_or_load_stats(self, dataset: 'WideCameraDataset', stats_path: str | None = None) -> Dict[str, Any]:
        if stats_path is not None:
            with open(stats_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            self.stats = loaded_data.get("statistics", loaded_data)
            self.normalized = True
            return self.stats

        phase_suffix = f"_{self.phase}" if self.phase else ""
        if len(dataset.task_name) > 0:
            save_path = f"{self.data_type}_action_normalization_stats{phase_suffix}{dataset.task_name[0]}.json"
        else:
            save_path = f"{self.data_type}_action_normalization_stats{phase_suffix}.json"
        if os.path.exists(save_path):
            print(f"检测到已存在的统计信息文件: {save_path}")
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                self.stats = loaded_data.get("statistics", loaded_data)
                self.normalized = True
                print("已成功加载动作和状态统计信息")
                return self.stats
            except Exception as e:
                print(f"加载统计信息失败: {e}，将重新计算...")

        print("开始计算动作和状态统计信息...")
        all_actions, all_proprios = dataset._collect_all_actions_and_proprios(phase=self.phase)

        if all_actions is not None and all_proprios is not None:
            print(f"收集到动作数据: {all_actions.shape}, 本体感觉状态数据: {all_proprios.shape}")
            self.stats = self._compute_detailed_stats(all_actions, all_proprios)
            self._save_stats(save_path)
            self.normalized = True
            print("动作和状态统计信息计算完成并已保存")
        else:
            print("警告: 未能收集到有效的动作或状态数据")
        return self.stats

    def _compute_detailed_stats(self, actions: np.ndarray, proprios: np.ndarray) -> Dict[str, Any]:
        stats = {'action': {}, 'proprio': {}}
        for i in range(actions.shape[1]):
            data = actions[:, i]
            stats['action'][f'dim_{i}'] = {
                'mean': float(np.mean(data)), 'std': float(np.std(data)),
                'min': float(np.min(data)), 'max': float(np.max(data)),
                'percentile_1': float(np.percentile(data, 1)), 'percentile_99': float(np.percentile(data, 99))
            }
        for i in range(proprios.shape[1]):
            data = proprios[:, i]
            stats['proprio'][f'dim_{i}'] = {
                'mean': float(np.mean(data)), 'std': float(np.std(data)),
                'min': float(np.min(data)), 'max': float(np.max(data)),
                'percentile_1': float(np.percentile(data, 1)), 'percentile_99': float(np.percentile(data, 99))
            }
        return stats

    def _save_stats(self, save_path: str):
        try:
            stats_with_config = {
                "normalization_config": {
                    "method": "1%-99% percentile min-max normalization",
                    "output_range": [0.0, 1.0],
                    "version": "1.4_wide_camera_subtask"
                },
                "statistics": self.stats
            }
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(stats_with_config, f, indent=2, ensure_ascii=False)
            print(f"动作和状态统计信息已保存到: {save_path}")
        except Exception as e:
            print(f"保存统计信息失败: {e}")

    def _normalize_min_max(self, data: torch.Tensor, key: str) -> torch.Tensor:
        if not self.normalized or not self.stats or key not in self.stats:
            return data
        normalized_data = data.clone()
        num_dims = data.shape[-1]
        for i in range(num_dims):
            # 跳过gripper维度（dim6和dim13），保持原始值
            if i in [6, 13]:
                continue
            dim_key = f'dim_{i}'
            if dim_key in self.stats[key]:
                stats_info = self.stats[key][dim_key]
                p1, p99 = stats_info['percentile_1'], stats_info['percentile_99']
                denom = max(p99 - p1, DELTA_EPS)
                normalized_data[..., i] = torch.clamp((data[..., i] - p1) / denom, 0, 1)
        return normalized_data
    def _normalize_zscore(self, data: torch.Tensor, key: str) -> torch.Tensor:
        if not self.normalized or not self.stats or key not in self.stats:
            return data
        normalized_data = data.clone()
        num_dims = data.shape[-1]
        for i in range(num_dims):
            # 跳过gripper维度（dim6和dim13），保持原始值
            if i in [6, 13]:
                continue
            dim_key = f'dim_{i}'
            if dim_key in self.stats[key]:
                stats_info = self.stats[key][dim_key]
                mean = stats_info['mean']
                std = stats_info['std']
                normalized_data[..., i] = (data[..., i] - mean) / (std + DELTA_EPS)

        return normalized_data

    def normalize_actions(self, actions: torch.Tensor, method: str = 'min_max') -> torch.Tensor:
        if method == 'min_max':
            return self._normalize_min_max(actions, 'action')
        elif method == 'zscore':
            return self._normalize_zscore(actions, 'action')
        else:
            raise ValueError(f"Invalid normalization method: {method}")

    def normalize_proprio(self, proprio: torch.Tensor, method: str = 'zscore') -> torch.Tensor:
        if method == 'min_max':
            return self._normalize_min_max(proprio, 'proprio')
        elif method == 'zscore':
            return self._normalize_zscore(proprio, 'proprio')
        else:
            raise ValueError(f"Invalid normalization method: {method}")

# ==================================================================================================
# WideCameraDataset Class (Final Refactoring)
# ==================================================================================================

class WideCameraDataset(Dataset):
    def __init__(self, data_dir, action_steps=10, image_transform=None,
                 normalize_actions=True,
                 cache_size=200, data_type='qpos', task_name=None,
                 enable_soft_route_labels: bool = False,
                 normalize_method: str = 'zscore',
                 use_shared_action_norm_stats: bool = True,
                 long_phase_ratio: float = 0.5,
                 drop_small_action_deltas: bool = True,
                 min_action_magnitude: float = SMALL_ACTION_EPS,
                 include_padding_in_mask: bool = False,
                 move_norm_stats_path: str | None = None,
                 operate_norm_stats_path: str | None = None,
                 shared_norm_stats_path: str | None = None):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.delta_from_first = self.data_type == 'qpos'
        self.action_steps = action_steps
        self.long_action_steps = action_steps
        self.task_name = task_name or []
        self.image_transform = image_transform
        self.cache_size = cache_size
        self.normalize_actions_flag = normalize_actions
        self.enable_soft_route_labels = enable_soft_route_labels
        self.use_shared_action_norm_stats = use_shared_action_norm_stats
        self.long_phase_ratio = float(max(0.0, min(1.0, long_phase_ratio)))
        self.drop_small_action_deltas = bool(drop_small_action_deltas)
        self.small_action_threshold = float(min_action_magnitude)
        self.include_padding_in_mask = bool(include_padding_in_mask)
        norm_data_type = f"{self.data_type}_delta" if self.delta_from_first else self.data_type

        phase_records = self._prepare_phase_records()
        self.subtasks = phase_records['subtasks']
        self.long_phase_entries = phase_records['long_phases']
        self.short_phase_entries = phase_records['short_phases']

        if not self.long_phase_entries and not self.short_phase_entries:
            raise ValueError("未找到任何 move/operate phase，请检查数据标注是否存在。")

        if self.use_shared_action_norm_stats:
            shared_normalizer = ActionNormalizer(norm_data_type, "")
            self.long_action_normalizer = shared_normalizer
            self.short_action_normalizer = shared_normalizer
        else:
            self.long_action_normalizer = ActionNormalizer(norm_data_type, "long")
            self.short_action_normalizer = ActionNormalizer(norm_data_type, "short")

        self.normalize_method = normalize_method
        if self.normalize_actions_flag:
            long_stats_path = shared_norm_stats_path if self.use_shared_action_norm_stats else move_norm_stats_path
            self.long_action_normalizer.compute_or_load_stats(self, stats_path=long_stats_path)
            if not self.use_shared_action_norm_stats:
                self.short_action_normalizer.compute_or_load_stats(self, stats_path=operate_norm_stats_path)

    def _prepare_phase_records(self):
        """Convert inclusive labels to [start, end) pools with at least two states per phase."""
        subtasks: List[Dict[str, Any]] = []
        long_phases: List[Dict[str, Any]] = []
        short_phases: List[Dict[str, Any]] = []

        data_glob_pattern = os.path.join(self.data_dir, "**", "data")
        data_dirs = glob.glob(data_glob_pattern, recursive=True)
        if self.task_name:
            data_dirs = [
                data_dir
                for data_dir in data_dirs
                if any(task_name in data_dir for task_name in self.task_name)
            ]

        for data_path in data_dirs:
            demo_base_dir = os.path.dirname(data_path)
            data_files = glob.glob(os.path.join(data_path, "*.hdf5"))

            for data_file in data_files:
                try:
                    episode_id_str = os.path.basename(data_file).replace(".hdf5", "").replace("episode", "")
                    episode_id = int(episode_id_str)
                except ValueError:
                    continue

                instruction_path = os.path.join(demo_base_dir, "instructions", f"episode{episode_id_str}.json")
                instruction_text = ""
                if os.path.exists(instruction_path):
                    with open(instruction_path, 'r', encoding='utf-8') as f:
                        instructions_data = json.load(f)
                        if 'seen' in instructions_data and instructions_data['seen']:
                            instruction_text = random.choice(instructions_data['seen'])

                label_path = os.path.join(
                    demo_base_dir,
                    "auto_labels_v2",
                    f"episode{episode_id_str}_phases_labels_thinking.json",
                )
                if not os.path.exists(label_path):
                    continue

                with open(label_path, 'r', encoding='utf-8') as f:
                    episode_subtasks = json.load(f)
                with h5py.File(data_file, 'r') as hf:
                    total_steps = hf['/joint_action/left_arm'].shape[0]

                for subtask_info in episode_subtasks:
                    subtask_start = int(subtask_info['start_frame_idx'])
                    subtask_end = int(subtask_info['end_frame_idx']) + 1
                    subtask_span = max(subtask_end - subtask_start, 1)
                    subtask_record = {
                        "hdf5_path": data_file,
                        "instruction": instruction_text,
                        "start_frame": subtask_start,
                        "end_frame": subtask_end,
                        "episode_id": episode_id,
                        "subtask_id": subtask_info.get("subtask"),
                        "subtask_span": subtask_span,
                        "phases": [],
                    }

                    for phase_idx, phase in enumerate(subtask_info.get("phases", [])):
                        phase_type = str(phase.get("phase_type", "")).lower()
                        if phase_type not in {"move", "operate"}:
                            continue

                        phase_start = max(0, int(phase.get("start_frame_idx", subtask_start)))
                        phase_end = min(total_steps, int(phase.get("end_frame_idx", subtask_end - 1)) + 1)
                        # One action requires two states; never borrow a state from the next phase.
                        if phase_end - phase_start < 2:
                            continue

                        phase_kind = "long" if phase_type == "move" else "short"
                        phase_entry = {
                            "hdf5_path": data_file,
                            "instruction": instruction_text,
                            "episode_id": episode_id,
                            "subtask_id": subtask_info.get("subtask"),
                            "phase_index": phase_idx,
                            "phase_type": phase_type,
                            "phase_kind": phase_kind,
                            "phase_start_frame": phase_start,
                            "phase_end_frame": phase_end,
                            "subtask_start_frame": subtask_start,
                            "subtask_end_frame": subtask_end,
                            "subtask_span": subtask_span,
                        }

                        subtask_record["phases"].append(phase_entry)
                        if phase_kind == "long":
                            long_phases.append(phase_entry)
                        else:
                            short_phases.append(phase_entry)

                    if subtask_record["phases"]:
                        subtasks.append(subtask_record)

        return {
            "subtasks": subtasks,
            "long_phases": long_phases,
            "short_phases": short_phases,
        }

    def __len__(self):
        return max(1, len(self.long_phase_entries) + len(self.short_phase_entries))


    def _pad_action_sequence(self, actions: torch.Tensor, length: int, *, pad_with_last: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dim = actions.shape[1]
        padded_actions = torch.zeros(length, action_dim, dtype=torch.float32)
        mask = torch.zeros(length, dtype=torch.bool)
        real_len = min(len(actions), length)
        padded_actions[:real_len] = actions[:real_len]
        mask[:real_len] = True
        if pad_with_last and real_len > 0 and real_len < length:
            padded_actions[real_len:] = padded_actions[real_len - 1]
        return padded_actions, mask

    def _filter_action_sequence_tensor(
        self,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.drop_small_action_deltas:
            return actions, mask
        if actions.numel() == 0 or mask is None or mask.numel() == 0:
            return actions, mask
        valid_len = int(mask.sum().item())
        if valid_len <= 0:
            return actions, mask

        valid_actions = actions[:valid_len]
        magnitude = valid_actions.abs().max(dim=-1).values
        keep = magnitude >= self.small_action_threshold
        if not torch.any(keep):
            keep = torch.zeros_like(magnitude, dtype=torch.bool)
            keep[0] = True

        filtered = valid_actions[keep]
        filtered_len = int(filtered.shape[0])
        if filtered_len == 0:
            filtered = valid_actions[:1]
            filtered_len = 1

        pad_value = filtered[-1] if filtered_len > 0 else torch.zeros_like(valid_actions[0])
        updated_actions = actions.clone()
        updated_actions[:filtered_len] = filtered
        if filtered_len < updated_actions.shape[0]:
            updated_actions[filtered_len:] = pad_value

        updated_mask = mask.clone()
        updated_mask[:] = self.include_padding_in_mask
        updated_mask[:filtered_len] = True
        return updated_actions, updated_mask

    def _filter_action_sequence_np(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.drop_small_action_deltas or actions.size == 0:
            keep = np.ones(actions.shape[0], dtype=bool)
            return actions, keep
        magnitude = np.max(np.abs(actions), axis=-1)
        keep = magnitude >= self.small_action_threshold
        if not np.any(keep):
            keep = np.zeros_like(magnitude, dtype=bool)
            keep[0] = True
        return actions[keep], keep

    def __getitem__(self, idx):
        phase_kind = self._sample_phase_kind()
        pool = self.long_phase_entries if phase_kind == "long" else self.short_phase_entries
        phase_entry = random.choice(pool)

        with h5py.File(phase_entry["hdf5_path"], 'r') as hf:
            sample = self._build_phase_sample(hf, phase_entry, phase_kind)

        normalizer = self.long_action_normalizer if phase_kind == "long" else self.short_action_normalizer
        if self.normalize_actions_flag:
            sample["action"] = normalizer.normalize_actions(sample["action"], self.normalize_method)
            sample["observation"]["state"] = normalizer.normalize_proprio(sample["observation"]["state"], self.normalize_method)

        if self.image_transform:
            sample["observation"]["image"] = {k: self.image_transform(v) for k, v in sample["observation"]["image"].items()}

        return sample

    def _sample_phase_kind(self) -> str:
        has_long = len(self.long_phase_entries) > 0
        has_short = len(self.short_phase_entries) > 0
        if has_long and has_short:
            return "long" if random.random() < self.long_phase_ratio else "short"
        return "long" if has_long else "short"

    def _sample_window_bounds(self, phase_start: int, phase_end: int, desired_actions: int, *, bias: str) -> Tuple[int, int]:
        phase_start = int(phase_start)
        phase_end = int(phase_end)

        min_states = max(2, desired_actions + 1)
        max_start = max(phase_start, phase_end - min_states)
        if max_start > phase_start:
            window_start = biased_randint(phase_start, max_start, min_tail_ratio=0.1, bias_towards=bias)
        else:
            window_start = phase_start

        window_end = min(phase_end, window_start + desired_actions + 1)
        return window_start, window_end

    def _build_phase_sample(self, hf, phase_entry: Dict[str, Any], phase_kind: str) -> Dict[str, Any]:
        desired_actions = self.long_action_steps if phase_kind == "long" else self.action_steps
        bias = "low" if phase_kind == "long" else "low"

        phase_start = int(phase_entry["phase_start_frame"])
        phase_end = int(phase_entry["phase_end_frame"])

        window_start, window_end = self._sample_window_bounds(phase_start, phase_end, desired_actions, bias=bias)

        phase_sample = self._build_qpos_phase_sample(
            hf, phase_entry, phase_kind, window_start, window_end, desired_actions
        )

        phase_sample["observation"]["instr"] = phase_entry["instruction"]
        phase_sample["route_label"] = torch.tensor(0.0 if phase_kind == "long" else 1.0, dtype=torch.float32)
        phase_sample["phase_type"] = phase_entry["phase_type"]

        if self.enable_soft_route_labels:
            subtask_start = phase_entry["subtask_start_frame"]
            subtask_span = max(phase_entry["subtask_span"], 1)
            progress = (window_start - subtask_start) / subtask_span
            phase_sample["route_soft_label"] = torch.tensor(float(np.clip(progress, 0.0, 1.0)), dtype=torch.float32)

        phase_sample["metadata"] = {
            "hdf5_path": phase_entry["hdf5_path"],
            "phase_kind": phase_kind,
            "phase_type": phase_entry["phase_type"],
            "phase_start": phase_start,
            "phase_end": phase_end,
            "window_start": window_start,
            "window_end": window_end,
            "episode_id": phase_entry.get("episode_id"),
            "subtask_id": phase_entry.get("subtask_id"),
            "phase_index": phase_entry.get("phase_index"),
        }
        return phase_sample

    def _build_qpos_phase_sample(self, hf, phase_entry, phase_kind, window_start, window_end, desired_actions):
        left_joint_dset = hf['/joint_action/left_arm']
        right_joint_dset = hf['/joint_action/right_arm']
        left_dim = left_joint_dset.shape[1] if len(left_joint_dset.shape) > 1 else 1
        right_dim = right_joint_dset.shape[1] if len(right_joint_dset.shape) > 1 else 1
        joint_dim = left_dim + right_dim + 2

        window_slice = slice(window_start, window_end)
        left_qp = _ensure_2d(np.asarray(left_joint_dset[window_slice]), left_dim)
        right_qp = _ensure_2d(np.asarray(right_joint_dset[window_slice]), right_dim)
        left_g = np.asarray(hf['/joint_action/left_gripper'][window_slice]).reshape(-1, 1)
        right_g = np.asarray(hf['/joint_action/right_gripper'][window_slice]).reshape(-1, 1)

        states_np = np.concatenate([left_qp, left_g, right_qp, right_g], axis=1).astype(np.float32, copy=False)
        if states_np.size == 0:
            states_np = np.zeros((1, joint_dim), dtype=np.float32)

        joint_dim = states_np.shape[1]
        current_state_tensor = torch.from_numpy(states_np[0].astype(np.float32, copy=False))

        if len(states_np) > 1:
            raw_actions = states_np[1:]
        else:
            raw_actions = np.zeros((1, joint_dim), dtype=np.float32)

        actions_tensor, action_mask = self._pad_action_sequence(
            torch.from_numpy(raw_actions.astype(np.float32, copy=False)),
            desired_actions,
            pad_with_last=False,
        )
        actions_tensor = _apply_qpos_delta_tensor(actions_tensor, action_mask, current_state_tensor)
        actions_tensor, action_mask = self._filter_action_sequence_tensor(actions_tensor, action_mask)
        action_dim_mask = torch.ones_like(actions_tensor, dtype=torch.bool)

        obs_images, _ = self._get_images_at_frame(hf, window_start)

        return {
            "observation": {
                "image": obs_images,
                "state": current_state_tensor,
            },
            "action": actions_tensor,
            "action_mask": action_mask,
            "action_loss_mask": action_mask.clone(),
            "action_dim_mask": action_dim_mask,
        }



    def _get_images_at_frame(self, hf, frame_idx):
        camera_datasets = {
            "head_camera": '/observation/head_camera/rgb',
            "left_camera": '/observation/left_camera/rgb',
            "right_camera": '/observation/right_camera/rgb',
        }
        images: Dict[str, Image.Image] = {}
        for cam_key, dataset_path in camera_datasets.items():
            jpeg_buffer = hf[dataset_path][frame_idx]
            with io.BytesIO(jpeg_buffer) as byte_stream:
                with Image.open(byte_stream) as img:
                    pil_image = img.convert("RGB")
                    pil_image.load()
                images[cam_key] = pil_image
        return images, frame_idx


    def _collect_all_actions_and_proprios(self, phase: str = "global"):
        """收集动作与本体数据用于归一化。"""
        long_actions, long_proprios = [], []
        short_actions, short_proprios = [], []

        if phase not in {"long", "short", "global"}:
            raise ValueError(f"Invalid phase argument: {phase}")

        target_phases = []
        if phase in {"long", "global"}:
            target_phases.extend(("long", entry) for entry in self.long_phase_entries)
        if phase in {"short", "global"}:
            target_phases.extend(("short", entry) for entry in self.short_phase_entries)

        for idx, (phase_kind, entry) in enumerate(target_phases):
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(target_phases)} phases for stats...")

            with h5py.File(entry["hdf5_path"], 'r') as hf:
                start = int(entry["phase_start_frame"])
                end = int(entry["phase_end_frame"])
                if end - start <= 1:
                    continue

                left_joint_dset = hf['/joint_action/left_arm']
                right_joint_dset = hf['/joint_action/right_arm']
                left_dim = left_joint_dset.shape[1] if len(left_joint_dset.shape) > 1 else 1
                right_dim = right_joint_dset.shape[1] if len(right_joint_dset.shape) > 1 else 1

                slice_obj = slice(start, end)
                left_qp = _ensure_2d(np.asarray(left_joint_dset[slice_obj]), left_dim)
                right_qp = _ensure_2d(np.asarray(right_joint_dset[slice_obj]), right_dim)
                left_g = np.asarray(hf['/joint_action/left_gripper'][slice_obj]).reshape(-1, 1)
                right_g = np.asarray(hf['/joint_action/right_gripper'][slice_obj]).reshape(-1, 1)
                qpos = np.concatenate([left_qp, left_g, right_qp, right_g], axis=1).astype(np.float32, copy=False)

                if len(qpos) == 0:
                    continue

                if len(qpos) > 1:
                    actions_arr = qpos[1:]
                else:
                    actions_arr = np.zeros((1, qpos.shape[1]), dtype=np.float32)

                if self.delta_from_first:
                    actions_arr = _apply_qpos_delta_np(actions_arr, qpos[0])
                actions_arr, keep_mask = self._filter_action_sequence_np(actions_arr)
                if keep_mask.size > 0:
                    state_keep = np.concatenate([[True], keep_mask])
                else:
                    state_keep = np.array([True], dtype=bool)
                qpos = qpos[state_keep]

                if phase_kind == "long":
                    long_actions.append(actions_arr.astype(np.float32, copy=False))
                    long_proprios.append(qpos.astype(np.float32, copy=False))
                else:
                    short_actions.append(actions_arr.astype(np.float32, copy=False))
                    short_proprios.append(qpos.astype(np.float32, copy=False))

        def _concat_or_none(collection: List[np.ndarray]):
            return np.concatenate(collection, axis=0) if collection else None

        if phase == "long":
            return _concat_or_none(long_actions), _concat_or_none(long_proprios)
        if phase == "short":
            return _concat_or_none(short_actions), _concat_or_none(short_proprios)

        combined_actions = [
            arr for arr in long_actions + short_actions if arr is not None and arr.size > 0
        ]
        combined_proprios = [
            arr for arr in long_proprios + short_proprios if arr is not None and arr.size > 0
        ]
        if not combined_actions or not combined_proprios:
            return None, None
        return _concat_or_none(combined_actions), _concat_or_none(combined_proprios)

def wide_camera_collate_fn(batch):
    """将单阶段样本批量化。"""
    if not batch:
        return {}

    collated = {
        "observation": {"image": {}, "state": None, "instr": []},
        "action": None,
        "action_mask": None,
        "route_label": None,
    }

    collated["action"] = torch.stack([item["action"] for item in batch])
    collated["action_mask"] = torch.stack([item["action_mask"] for item in batch])
    collated["route_label"] = torch.stack([item["route_label"] for item in batch])
    if "action_loss_mask" in batch[0]:
        collated["action_loss_mask"] = torch.stack([item["action_loss_mask"] for item in batch])
    if "action_dim_mask" in batch[0]:
        collated["action_dim_mask"] = torch.stack([item["action_dim_mask"] for item in batch])

    if "route_soft_label" in batch[0]:
        collated["route_soft_label"] = torch.stack([item["route_soft_label"] for item in batch])

    collated["observation"]["state"] = torch.stack([item["observation"]["state"] for item in batch])
    collated["observation"]["instr"] = [item["observation"]["instr"] for item in batch]

    first_obs_images = batch[0]["observation"]["image"]
    for cam_key in first_obs_images.keys():
        collated["observation"]["image"][cam_key] = torch.stack(
            [item["observation"]["image"][cam_key] for item in batch]
        )

    collated["phase_type"] = [item.get("phase_type", "") for item in batch]
    collated["metadata"] = [item.get("metadata", {}) for item in batch]

    return collated
