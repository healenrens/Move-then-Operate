"""One integration test for LeRobot storage through normalized training batches.

Run with: .venv/bin/python -m unittest discover -s tests -p test_data_pipeline.py
The MP4 fixture uses MPEG-4; production AV1 decoding needs the real-data check.
"""

from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import unittest

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mto.compute_norm_stats import compute_statistics
from mto.dataset import CAMERAS, PhaseDataset, wide_camera_collate_fn
from mto.labels import label_path, load_phase_labels
from mto.lerobot_reader import LeRobotReader, list_episodes
from mto.normalization import JOINT_INDICES, QposNormalizer
from mto.train import (WideCameraTrainConfig, _build_observation_and_actions, _create_train_config,
                       archive_normalization, record_run, resolve_resume_config)


FPS = 30
LENGTH = 8
HORIZON = 3


def _color(task_number, camera_number, video_frame):
    return np.array([20 + video_frame * 8, 40 + camera_number * 70, 60 + task_number * 120], np.uint8)


def _write_video(path, task_number, camera_number, episode_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 1_000_000
        stream.codec_context.gop_size = 6
        for video_frame in range(episode_count * LENGTH):
            rgb = np.broadcast_to(_color(task_number, camera_number, video_frame), (32, 32, 3)).copy()
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = video_frame
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_tasks(path, task_number, instruction):
    # An unrelated first string column makes guessing the instruction column fail.
    table = pa.table({"unrelated": ["not the instruction"], "task_index": [7 + task_number],
                      "instruction_index": [instruction]})
    pandas_metadata = {
        "index_columns": ["instruction_index"], "column_indexes": [],
        "columns": [
            {"name": "unrelated", "field_name": "unrelated", "pandas_type": "unicode",
             "numpy_type": "object", "metadata": None},
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
            {"name": "instruction", "field_name": "instruction_index", "pandas_type": "unicode",
             "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.3",
    }
    table = table.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode()})
    pq.write_table(table, path)


def _write_dataset(collection, labels_root, task_number, episode_count):
    task = f"task_{task_number}"
    dataset_root = collection / task / "demo_clean"
    (dataset_root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/rows-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/clip-{file_index:03d}.mp4",
    }
    (dataset_root / "meta/info.json").write_text(json.dumps(info))
    instruction = f"Operate task {task_number}."
    _write_tasks(dataset_root / "meta/tasks.parquet", task_number, instruction)
    expected, metadata_rows, data_shards = {}, [], {}
    for episode_index in range(episode_count):
        first = 100 + episode_index * LENGTH if episode_index < 2 else 700
        chunk, file = (3, 2) if episode_index < 2 else (9, 4)
        t = np.arange(LENGTH + 1, dtype=np.float32)[:, None]
        states = task_number * 3 + episode_index * 0.4 + t**2 * 0.125 + np.arange(14, dtype=np.float32) * 0.01
        states[:, 6] = np.arange(LENGTH + 1) / 10
        states[:, 13] = (LENGTH - np.arange(LENGTH + 1)) / 10
        states[2] = 0  # A genuine all-zero absolute action target, not padding.
        actions = states[1:].copy()
        identity = (f"{task}/demo_clean", episode_index)
        expected[identity] = {"state": states[:-1], "action": actions, "first": first,
                              "task_number": task_number, "instruction": instruction}
        rows = [
            {"observation.state": states[i].tolist(), "action": actions[i].tolist(),
             "timestamp": float(np.float32(i / FPS)), "frame_index": i,
             "episode_index": episode_index, "index": first + i, "task_index": 7 + task_number}
            for i in range(LENGTH)
        ]
        data_shards.setdefault((chunk, file), []).extend(reversed(rows))
        metadata = {"episode_index": episode_index, "length": LENGTH, "dataset_from_index": first,
                    "dataset_to_index": first + LENGTH, "data/chunk_index": chunk, "data/file_index": file}
        for camera in CAMERAS.values():
            metadata.update({f"videos/{camera}/chunk_index": 6, f"videos/{camera}/file_index": 11,
                             f"videos/{camera}/from_timestamp": episode_index * LENGTH / FPS,
                             f"videos/{camera}/to_timestamp": (episode_index + 1) * LENGTH / FPS})
        metadata_rows.append(metadata)
        labels = [{"start_frame_idx": 0, "end_frame_idx": LENGTH - 1, "phases": [
            {"phase_type": "move", "start_frame_idx": 0, "end_frame_idx": 3},
            {"phase_type": "operate", "start_frame_idx": 4, "end_frame_idx": 7},
        ]}]
        path = label_path(dataset_root, identity[0], episode_index, labels_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels))
    for (chunk, file), rows in data_shards.items():
        path = dataset_root / info["data_path"].format(chunk_index=chunk, file_index=file)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
    # Metadata is split independently of data/video shards, with nonzero shard IDs.
    for shard_number, rows in enumerate((metadata_rows[::2], metadata_rows[1::2])):
        if rows:
            path = dataset_root / f"meta/episodes/chunk-004/file-{shard_number + 7:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
    for camera_number, camera in enumerate(CAMERAS.values()):
        path = dataset_root / info["video_path"].format(video_key=camera, chunk_index=6, file_index=11)
        _write_video(path, task_number, camera_number, episode_count)
    return expected


def _expected_window(record, anchor, end):
    n = min(HORIZON, end - anchor)
    actions = record["action"][anchor:anchor + n].copy()
    actions[:, JOINT_INDICES] -= record["state"][anchor, JOINT_INDICES]
    return actions


class _Tokenizer:
    def tokenize(self, prompt):
        return np.array([1, len(prompt), 0], np.int32), np.array([True, True, False])


class DataPipelineIntegrationTest(unittest.TestCase):
    def test_lerobot_to_per_expert_normalized_training_batch(self):
        with tempfile.TemporaryDirectory(prefix="mto-data-pipeline-") as directory:
            root = Path(directory)
            collection, labels_root = root / "data", root / "labels"
            expected = _write_dataset(collection, labels_root, 0, 3)
            expected.update(_write_dataset(collection, labels_root, 1, 1))
            reader = LeRobotReader(collection)
            self.assertEqual([(ep.relative_id, ep.episode_index) for ep in reader.episodes], sorted(expected))
            single_root = collection / "task_0/demo_clean"
            self.assertEqual([ep.relative_id for ep in list_episodes(single_root)], ["task_0/demo_clean"] * 3)
            self.assertEqual(len(list_episodes(collection / "task_0", split_names=("demo_clean",))), 3)
            self.assertEqual(len(list_episodes(collection, task_names=("task_1",))), 1)

            for episode in reader.episodes:
                record = expected[(episode.relative_id, episode.episode_index)]
                rows = reader.read_rows(episode)
                np.testing.assert_array_equal(rows["index"], np.arange(record["first"], record["first"] + LENGTH))
                np.testing.assert_array_equal(rows["frame_index"], np.arange(LENGTH))
                np.testing.assert_array_equal(rows["observation.state"], record["state"])
                np.testing.assert_array_equal(rows["action"], record["action"])
                np.testing.assert_array_equal(reader.read_rows(episode, 2, 5)["action"], record["action"][2:5])
                self.assertEqual(reader.get_instruction(episode), record["instruction"])
                labels, error = load_phase_labels(
                    label_path(episode.dataset_root, episode.relative_id, episode.episode_index, labels_root), LENGTH)
                self.assertIsNone(error)
                self.assertEqual(labels[0]["phases"][1]["start_frame_idx"], 4)
                requested = [7, 0, 4, 4]
                for camera_number, camera in enumerate(CAMERAS.values()):
                    images = reader.read_rgb(episode, camera, requested)
                    for local_frame, image in zip(requested, images, strict=True):
                        self.assertEqual(image.mode, "RGB")
                        expected_color = _color(record["task_number"], camera_number,
                                                episode.episode_index * LENGTH + local_frame)
                        np.testing.assert_allclose(np.asarray(image).mean(axis=(0, 1)), expected_color, atol=3)

            dataset = PhaseDataset(collection, labels_root=labels_root, action_steps=HORIZON,
                                   normalize_actions=False)
            self.assertEqual(dataset.manifest["phase_counts"], {"move": 4, "operate": 4})
            self.assertEqual(dataset.manifest["missing_labels"], [])
            self.assertEqual(dataset.manifest["invalid_labels"], [])
            for pool in dataset.pools.values():
                for entry in pool:
                    episode = entry["episode"]
                    record = expected[(episode.relative_id, episode.episode_index)]
                    for anchor in range(entry["start"], entry["end"]):
                        raw = dataset.raw_window(entry, anchor)
                        target = _expected_window(record, anchor, entry["end"])
                        n = len(target)
                        np.testing.assert_array_equal(raw["state"], record["state"][anchor])
                        np.testing.assert_array_equal(raw["action"][:n], target)
                        np.testing.assert_array_equal(raw["action"][n:], 0)
                        np.testing.assert_array_equal(raw["mask"], np.arange(HORIZON) < n)
                        # Add the one chunk anchor back, never a cumulative sum.
                        absolute = raw["action"][:n].copy()
                        absolute[:, JOINT_INDICES] += raw["state"][JOINT_INDICES]
                        np.testing.assert_allclose(absolute, record["action"][anchor:anchor + n], atol=1e-6)
                        if anchor == 1:
                            self.assertTrue(raw["mask"][0])
                            np.testing.assert_array_equal(absolute[0], np.zeros(14))

            statistics_by_phase = {}
            for phase in ("move", "operate"):
                statistics, count = compute_statistics(dataset, phase, num_samples=48, seed=17)
                rng = random.Random(17)
                actions, states = [], []
                for _ in range(48):
                    raw = dataset.sample_raw(rng, phase)
                    meta = raw["metadata"]
                    record = expected[(meta["dataset"], meta["episode_index"])]
                    actions.append(_expected_window(record, meta["window_start"], meta["phase_end"]))
                    states.append(record["state"][meta["window_start"]])
                actions, states = np.concatenate(actions), np.stack(states)
                self.assertEqual(count, len(actions))
                self.assertLess(count, 48 * HORIZON)  # Padded slots never enter statistics.
                for key, values in (("action", actions), ("proprio", states)):
                    for dim in range(14):
                        stats = statistics[key][f"dim_{dim}"]
                        self.assertAlmostEqual(stats["mean"], float(values[:, dim].mean()), places=6)
                        self.assertAlmostEqual(stats["std"], float(values[:, dim].std()), places=6)
                        self.assertAlmostEqual(stats["percentile_1"], float(np.percentile(values[:, dim], 1)), places=6)
                path = root / f"{phase}.json"
                path.write_text(json.dumps({"statistics": statistics}))
                normalizer = QposNormalizer(path)
                normalized = normalizer.normalize(actions, "action")
                np.testing.assert_allclose(normalizer.unnormalize_actions(normalized), actions, atol=2e-6)
                np.testing.assert_array_equal(normalized[:, [6, 13]], actions[:, [6, 13]])
                statistics_by_phase[phase] = statistics
            self.assertNotEqual(statistics_by_phase["move"]["action"]["dim_0"]["mean"],
                                statistics_by_phase["operate"]["action"]["dim_0"]["mean"])

            normalized_dataset = PhaseDataset(
                collection, labels_root=labels_root, action_steps=HORIZON,
                move_norm_stats_path=root / "move.json", operate_norm_stats_path=root / "operate.json")
            samples = []
            random.seed(23)
            for route, phase in enumerate(("move", "operate")):
                normalized_dataset.long_phase_ratio = 1.0 - route
                sample = normalized_dataset[0]
                self.assertEqual(sample["route_label"].item(), route)
                meta = sample["metadata"]
                record = expected[(meta["dataset"], meta["episode_index"])]
                target = _expected_window(record, meta["window_start"], meta["phase_end"])
                mask = sample["action_mask"].numpy()
                normalizer = normalized_dataset.normalizers[phase]
                recovered = normalizer.unnormalize_actions(sample["action"].numpy()[mask])
                np.testing.assert_allclose(recovered, target, atol=2e-6)
                np.testing.assert_array_equal(sample["action"].numpy()[~mask], 0)
                np.testing.assert_allclose(sample["observation"]["state"].numpy(),
                                           normalizer.normalize_state(record["state"][meta["window_start"]]))
                samples.append(sample)
            batch = wide_camera_collate_fn(samples)
            observation, actions = _build_observation_and_actions(batch, _Tokenizer(), 32, HORIZON)
            self.assertEqual(actions.shape, (2, HORIZON, 32))
            self.assertEqual(observation.state.shape, (2, 32))
            np.testing.assert_array_equal(observation.route_labels, [0, 1])
            np.testing.assert_array_equal(observation.action_loss_mask, batch["action_mask"].numpy())
            np.testing.assert_array_equal(observation.action_dim_mask[:, :14], True)
            np.testing.assert_array_equal(observation.action_dim_mask[:, 14:], False)
            np.testing.assert_array_equal(actions[:, :, 14:], 0)
            np.testing.assert_array_equal(observation.state[:, 14:], 0)
            for image in observation.images.values():
                self.assertEqual(image.shape, (2, 224, 224, 3))
                self.assertGreaterEqual(image.min(), -1)
                self.assertLessEqual(image.max(), 1)

            # The retained HDF5 backend reaches the same next-state targets.
            legacy_root = root / "legacy/task_0/demo_clean"
            (legacy_root / "data").mkdir(parents=True)
            reference = expected[("task_0/demo_clean", 0)]
            with h5py.File(legacy_root / "data/episode0.hdf5", "w") as hf:
                for key, columns in (("left_arm", slice(0, 6)), ("left_gripper", slice(6, 7)),
                                     ("right_arm", slice(7, 13)), ("right_gripper", slice(13, 14))):
                    hf.create_dataset(f"joint_action/{key}", data=reference["state"][:, columns])
            legacy = PhaseDataset(root / "legacy", data_format="hdf5", labels_root=labels_root,
                                  action_steps=HORIZON, normalize_actions=False)
            for pool in legacy.pools.values():
                for entry in pool:
                    for anchor in range(entry["start"], entry["end"]):
                        raw = legacy.raw_window(entry, anchor)
                        np.testing.assert_array_equal(raw["action"][raw["mask"]],
                                                      _expected_window(reference, anchor, entry["end"]))

            # Saved contracts survive CLI defaults during resume, with separate assets.
            cfg = WideCameraTrainConfig(data_root=str(collection), labels_root=str(labels_root),
                                        exp_name="fixture", checkpoint_base_dir=str(root / "runs"),
                                        move_norm_stats_path=str(root / "move.json"),
                                        operate_norm_stats_path=str(root / "operate.json"),
                                        action_horizon=HORIZON, training_mode="lora", normalize_method="min_max")
            train_cfg = _create_train_config(cfg)
            train_cfg.checkpoint_dir.mkdir(parents=True)
            archive_normalization(cfg, train_cfg.checkpoint_dir, resuming=False)
            record_run(cfg, train_cfg, dataset, resuming=False)
            request = WideCameraTrainConfig(data_root="not_the_saved_data", exp_name="fixture",
                                            checkpoint_base_dir=str(root / "runs"), num_steps=20000)
            resumed = resolve_resume_config(request)
            self.assertEqual((resumed.action_horizon, resumed.normalize_method, resumed.training_mode),
                             (HORIZON, "min_max", "lora"))
            self.assertEqual(resumed.data_root, str(collection))
            self.assertEqual(resumed.num_steps, 20000)
            archive_normalization(resumed, train_cfg.checkpoint_dir, resuming=True)
            for phase in ("move", "operate"):
                self.assertEqual(json.loads((train_cfg.checkpoint_dir / "assets" / f"{phase}.json").read_text()),
                                 json.loads((root / f"{phase}.json").read_text()))


if __name__ == "__main__":
    unittest.main()
