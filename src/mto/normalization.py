"""Independent joint14 statistics; grippers always remain in absolute coordinates."""
import json
from pathlib import Path

import numpy as np

JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
ACTION_REPRESENTATION = "joint14_delta_joint_absolute_gripper"


class QposNormalizer:
    def __init__(self, path, method="zscore"):
        payload = json.loads(Path(path).read_text())
        self.metadata = payload.get("normalization_config", {})
        statistics = payload.get("statistics", payload)
        self.method = method
        self.offset, self.scale = {}, {}
        for key in ("action", "proprio"):
            dims = [statistics[key][f"dim_{i}"] for i in JOINT_INDICES]
            if method == "zscore":
                offset = [dim["mean"] for dim in dims]
                scale = np.array([dim["std"] for dim in dims], np.float32) + 1e-6
            else:
                offset = [dim["percentile_1"] for dim in dims]
                scale = np.maximum(np.array([dim["percentile_99"] for dim in dims]) - offset, 1e-6)
            self.offset[key] = np.asarray(offset, np.float32)
            self.scale[key] = np.asarray(scale, np.float32)

    def normalize(self, values, key):
        output = np.array(values, dtype=np.float32, copy=True)
        joints = (output[..., JOINT_INDICES] - self.offset[key]) / self.scale[key]
        if self.method == "min_max":
            joints = np.clip(joints, 0.0, 1.0)
        output[..., JOINT_INDICES] = joints
        return output

    def normalize_state(self, state):
        return self.normalize(state, "proprio")

    def unnormalize_actions(self, actions):
        output = np.array(actions, dtype=np.float32, copy=True)
        output[..., JOINT_INDICES] = output[..., JOINT_INDICES] * self.scale["action"] + self.offset["action"]
        return output


def detailed_statistics(actions, states):
    result = {}
    for key, values in (("action", actions), ("proprio", states)):
        result[key] = {
            f"dim_{i}": {
                "mean": float(np.mean(values[:, i])), "std": float(np.std(values[:, i])),
                "min": float(np.min(values[:, i])), "max": float(np.max(values[:, i])),
                "percentile_1": float(np.percentile(values[:, i], 1)),
                "percentile_99": float(np.percentile(values[:, i], 99)),
            } for i in range(14)
        }
    return result
