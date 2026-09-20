"""Read LeRobot v3 episodes without depending on the LeRobot training package.

Dataset indices, episode frame indices, and video presentation timestamps are
kept separate. Returned action rows retain their original temporal alignment.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EpisodeRef:
    dataset_root: Path
    relative_id: str
    episode_index: int
    length: int
    fps: float
    metadata: dict
    info: dict


def list_episodes(
    data_root: str | Path,
    task_names: Sequence[str] = (),
    split_names: Sequence[str] = (),
) -> list[EpisodeRef]:
    """Enumerate every metadata shard below a dataset, task, or collection root."""
    data_root = Path(data_root).expanduser().resolve()
    info_paths = sorted(data_root.glob("**/meta/info.json"))
    episodes = []
    for info_path in info_paths:
        dataset_root = info_path.parent.parent
        task, split = dataset_root.parent.name, dataset_root.name
        if task_names and task not in task_names:
            continue
        if split_names and split not in split_names:
            continue
        info = json.loads(info_path.read_text())
        assert info["codebase_version"] == "v3.0", f"{dataset_root}: expected LeRobot v3.0"
        # Identity is independent of whether the caller supplied the whole
        # collection, a task directory, or this task/split directory itself.
        relative_id = f"{task}/{split}"
        for metadata_path in sorted((dataset_root / "meta/episodes").rglob("*.parquet")):
            for metadata in pq.read_table(metadata_path).to_pylist():
                episodes.append(
                    EpisodeRef(
                        dataset_root=dataset_root,
                        relative_id=relative_id,
                        episode_index=int(metadata["episode_index"]),
                        length=int(metadata["length"]),
                        fps=float(info["fps"]),
                        metadata=metadata,
                        info=info,
                    )
                )
    return sorted(episodes, key=lambda ep: (ep.relative_id, ep.episode_index))


class LeRobotReader:
    """Read episode rows and RGB frames with a bounded per-worker row cache."""

    def __init__(
        self,
        data_root: str | Path,
        task_names: Sequence[str] = (),
        split_names: Sequence[str] = (),
    ):
        self.episodes = list_episodes(data_root, task_names, split_names)
        self._row_cache: OrderedDict[tuple[Path, int], dict[str, np.ndarray]] = OrderedDict()
        self._row_cache_limit = 8
        self._cache_pid = os.getpid()
        self._instructions: dict[Path, dict[int, str]] = {}

    def get_instruction(self, ep: EpisodeRef) -> str:
        """Resolve the episode's task_index using the stored pandas string index."""
        if ep.dataset_root not in self._instructions:
            task_path = ep.dataset_root / "meta/tasks.parquet"
            tasks = pq.read_table(task_path)
            pandas_metadata = tasks.schema.pandas_metadata
            assert pandas_metadata is not None, f"{task_path}: missing pandas index metadata"
            index_columns = [
                name
                for name in pandas_metadata["index_columns"]
                if isinstance(name, str)
                and (
                    pa.types.is_string(tasks.schema.field(name).type)
                    or pa.types.is_large_string(tasks.schema.field(name).type)
                )
            ]
            assert len(index_columns) == 1, f"{task_path}: expected one instruction string index"
            self._instructions[ep.dataset_root] = dict(
                zip(tasks["task_index"].to_pylist(), tasks[index_columns[0]].to_pylist(), strict=True)
            )
        task_indices = np.unique(self.read_rows(ep)["task_index"])
        assert len(task_indices) == 1, f"{ep.relative_id}/{ep.episode_index}: expected one task per episode"
        return self._instructions[ep.dataset_root][int(task_indices[0])]

    def _episode_rows(self, ep: EpisodeRef) -> dict[str, np.ndarray]:
        # A forked/spawned DataLoader worker starts its own bounded cache.
        if self._cache_pid != os.getpid():
            self._row_cache.clear()
            self._cache_pid = os.getpid()
        key = (ep.dataset_root, ep.episode_index)
        if key in self._row_cache:
            self._row_cache.move_to_end(key)
            return self._row_cache[key]

        metadata = ep.metadata
        first = int(metadata["dataset_from_index"])
        stop = int(metadata["dataset_to_index"])
        data_path = ep.dataset_root / ep.info["data_path"].format(
            chunk_index=int(metadata["data/chunk_index"]),
            file_index=int(metadata["data/file_index"]),
            episode_index=ep.episode_index,
        )
        # Parquet predicates prune unrelated row groups before loading rows;
        # dataset_from_index is a global value, never a shard-local iloc.
        table = pq.read_table(
            data_path,
            filters=[
                ("episode_index", "=", ep.episode_index),
                ("index", ">=", first),
                ("index", "<", stop),
            ],
        ).sort_by("frame_index")
        identity = f"{ep.relative_id}/{ep.episode_index}"
        assert stop - first == ep.length, f"{identity}: metadata index range disagrees with length"
        assert table.num_rows == ep.length, f"{identity}: actual row count disagrees with length"
        rows = {}
        for name in table.column_names:
            column = table[name]
            if name in ("observation.state", "action"):
                rows[name] = np.asarray(column.to_pylist(), dtype=np.float32)
            else:
                rows[name] = column.to_numpy(zero_copy_only=False)
        assert np.array_equal(rows["frame_index"], np.arange(ep.length)), (
            f"{identity}: frame_index must cover the consecutive episode-local range"
        )
        assert np.array_equal(rows["index"], np.arange(first, stop)), (
            f"{identity}: global index must cover the consecutive metadata range"
        )
        self._row_cache[key] = rows
        if len(self._row_cache) > self._row_cache_limit:
            self._row_cache.popitem(last=False)
        return rows

    def read_rows(self, ep: EpisodeRef, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        """Return episode-local [start, stop) rows; callers copy arrays before editing."""
        stop = ep.length if stop is None else stop
        assert 0 <= start <= stop <= ep.length, f"{ep.relative_id}/{ep.episode_index}: invalid local row range"
        return {name: values[start:stop] for name, values in self._episode_rows(ep).items()}

    def read_rgb(
        self,
        ep: EpisodeRef,
        camera_key: str,
        local_frame_indices: Sequence[int],
    ) -> list[Image.Image]:
        """Decode RGB images in requested order using absolute video PTS.

        camera_key is the stored feature name, e.g. observation.images.cam_high.
        Repeated or out-of-order local frame indices are supported.
        """
        if len(local_frame_indices) == 0:
            return []
        indices = np.asarray(local_frame_indices)
        assert np.issubdtype(indices.dtype, np.integer), "RGB frame indices must be integers"
        assert np.all((indices >= 0) & (indices < ep.length)), (
            f"{ep.relative_id}/{ep.episode_index}: RGB frame indices are outside the episode"
        )
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        prefix = f"videos/{camera_key}"
        path = ep.dataset_root / ep.info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(ep.metadata[f"{prefix}/chunk_index"]),
            file_index=int(ep.metadata[f"{prefix}/file_index"]),
            episode_index=ep.episode_index,
        )
        timestamps = (
            float(ep.metadata[f"{prefix}/from_timestamp"])
            + self._episode_rows(ep)["timestamp"][unique_indices].astype(np.float64)
        )
        images = _decode_rgb(path, timestamps, ep.fps)
        return [images[index] for index in inverse]


def _decode_rgb(path: Path, timestamps: np.ndarray, fps: float) -> list[Image.Image]:
    """Seek to a preceding keyframe, then choose the nearest presentation frame."""
    order = np.argsort(timestamps, kind="stable")
    images: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        seek_pts = math.floor(float(timestamps[order[0]]) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        frames = iter(container.decode(stream))
        frame = next(frames)
        current = (float(frame.pts * frame.time_base), frame)
        previous = None
        for index in order:
            target = float(timestamps[index])
            while current is not None and current[0] < target:
                previous = current
                frame = next(frames, None)
                current = None if frame is None else (float(frame.pts * frame.time_base), frame)
            candidates = [candidate for candidate in (previous, current) if candidate is not None]
            pts, selected = min(candidates, key=lambda candidate: abs(candidate[0] - target))
            assert abs(pts - target) <= 0.5 / fps + 1e-6, (
                f"{path}: no video frame at {target:.9f}s; nearest PTS is {pts:.9f}s"
            )
            images[int(index)] = selected.to_image().convert("RGB")
    return [images[index] for index in range(len(timestamps))]
