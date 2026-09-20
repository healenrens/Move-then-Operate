"""Shared phase-label paths and episode-local temporal contracts."""

import json
from pathlib import Path


def label_path(
    dataset_root: Path,
    relative_id: str,
    episode_index: int,
    labels_root: Path | None = None,
) -> Path:
    directory = Path(dataset_root) if labels_root is None else Path(labels_root) / relative_id
    return directory / "auto_labels_v2" / f"episode{episode_index}_phases_labels_thinking.json"


def label_metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def validate_phase_labels(labels: object, length: int) -> str | None:
    """Validate inclusive state-frame intervals covering one complete episode.

    Single-state phases and episodes containing only one phase type are valid.
    Sampling action transitions from these intervals is the dataset's concern.
    """
    if length < 1:
        return "Episode length must be positive."
    if not isinstance(labels, list) or not labels:
        return "Labels must be a nonempty subtask array."

    next_subtask_start = 0
    for subtask_number, subtask in enumerate(labels, start=1):
        if not isinstance(subtask, dict):
            return f"Subtask {subtask_number} must be an object."
        start = subtask.get("start_frame_idx")
        end = subtask.get("end_frame_idx")
        if type(start) is not int or type(end) is not int:
            return f"Subtask {subtask_number} frame indices must be integers, not booleans or strings."
        if not 0 <= start <= end < length:
            return f"Subtask {subtask_number} interval [{start}, {end}] is outside [0, {length - 1}]."
        if start != next_subtask_start:
            return f"Subtask {subtask_number} must start at {next_subtask_start}, got {start}."

        phases = subtask.get("phases")
        if not isinstance(phases, list) or not phases:
            return f"Subtask {subtask_number} must contain a nonempty phases array."
        next_phase_start = start
        for phase_number, phase in enumerate(phases, start=1):
            location = f"Subtask {subtask_number}, phase {phase_number}"
            if not isinstance(phase, dict):
                return f"{location} must be an object."
            if phase.get("phase_type") not in ("move", "operate"):
                return f"{location} phase_type must be move or operate."
            phase_start = phase.get("start_frame_idx")
            phase_end = phase.get("end_frame_idx")
            if type(phase_start) is not int or type(phase_end) is not int:
                return f"{location} frame indices must be integers, not booleans or strings."
            if not start <= phase_start <= phase_end <= end:
                return f"{location} interval [{phase_start}, {phase_end}] exceeds its subtask [{start}, {end}]."
            if phase_start != next_phase_start:
                return f"{location} must start at {next_phase_start}, got {phase_start}."
            next_phase_start = phase_end + 1
        if next_phase_start != end + 1:
            return f"Subtask {subtask_number} phases must end at {end}, got {next_phase_start - 1}."
        next_subtask_start = end + 1

    if next_subtask_start != length:
        return f"Last subtask must end at {length - 1}, got {next_subtask_start - 1}."
    return None


def load_phase_labels(path: Path, length: int) -> tuple[list[dict], str | None]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            labels = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [], f"Cannot read labels {path}: {error}"
    error = validate_phase_labels(labels, length)
    return ([], error) if error else (labels, None)
