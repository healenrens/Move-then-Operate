"""One offline annotation-pipeline regression test; no API requests are sent."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError
from PIL import Image

from mto import auto_label
from mto.labels import label_metadata_path, label_path, load_phase_labels, validate_phase_labels


class LabelPipelineTest(unittest.TestCase):
    def test_annotation_pipeline(self):
        indices = auto_label.sample_frame_indices(720, 30, 5, 64)
        self.assertEqual((indices[0], indices[-1], len(indices)), (0, 719, 64))
        self.assertEqual(indices, sorted(set(indices)))
        gaps = [right - left for left, right in zip(indices, indices[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)
        self.assertEqual(auto_label.sample_frame_indices(1, 30, 5, 64), [0])

        labels = [{
            "subtask": 1, "subtask_description": "Pick up the bottle.", "primary_arm": "left",
            "start_frame_idx": 0, "end_frame_idx": 719, "target_object_name": "bottle",
            "target_object_axis": [0.4, 0.5], "left_gripper_end_axis": [0.2, 0.6],
            "right_gripper_end_axis": [-1, -1], "phases": [
                {"phase_type": "move", "phase_description": "Approach.", "start_frame_idx": 0, "end_frame_idx": 99},
                {"phase_type": "operate", "phase_description": "Pick up.", "start_frame_idx": 100, "end_frame_idx": 719},
            ],
        }]
        self.assertIsNone(validate_phase_labels(labels, 720))
        for bad_start in (99, 101, True, 100.0, "100", -1):
            broken = deepcopy(labels)
            broken[0]["phases"][1]["start_frame_idx"] = bad_start
            self.assertTrue(auto_label.validate_subtasks(broken, 720)[1])
        for start, end in ((1, 719), (0, 718), (0, 720)):
            broken = deepcopy(labels)
            broken[0]["start_frame_idx"], broken[0]["end_frame_idx"] = start, end
            self.assertIsNotNone(validate_phase_labels(broken, 720))
        pure_operate = deepcopy(labels)
        pure_operate[0]["end_frame_idx"] = 0
        pure_operate[0]["phases"] = [{
            "phase_type": "operate", "phase_description": "Hold.", "start_frame_idx": 0, "end_frame_idx": 0,
        }]
        self.assertFalse(auto_label.validate_subtasks(pure_operate, 1)[1])

        args = argparse.Namespace(data_format="lerobot", camera_key="observation.images.cam_high",
                                  sample_fps=5.0, max_frames=64, model_id="mock-model", base_url="http://localhost",
                                  max_new_tokens=2048, max_attempts=3, concurrency=1, resume=True, api_key=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "source/task_a/demo_clean"
            output = label_path(dataset, "task_a/demo_clean", 0, root / "labels")
            other = label_path(root / "source/task_b/demo_clean", "task_b/demo_clean", 0, root / "labels")
            self.assertNotEqual(output, other)
            self.assertEqual(output.relative_to(root / "labels").as_posix(),
                             "task_a/demo_clean/auto_labels_v2/episode0_phases_labels_thinking.json")
            self.assertEqual(label_path(dataset, "task_a/demo_clean", 0).parent, dataset / "auto_labels_v2")
            metadata = auto_label.build_metadata(args, "task_a/demo_clean", 0, 720, 30.0,
                                                 "Pick up the bottle.", {"dataset_root": str(dataset)})
            job = auto_label.AnnotationJob(output, metadata, indices, video_path="mock.mp4")
            frame = Image.new("RGB", (8, 8))
            connection_error = APIConnectionError(request=httpx.Request("POST", "http://localhost"))
            with patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, frame)), \
                    patch.object(auto_label, "run_inference",
                                 side_effect=[connection_error, "[invalid JSON]", json.dumps(labels)]) as infer, \
                    redirect_stdout(io.StringIO()):
                result = auto_label.process_job(object(), args, job, None)
            self.assertEqual((result["status"], infer.call_count), ("annotated", 3))
            self.assertEqual(load_phase_labels(output, 720), (labels, None))
            self.assertIsNone(auto_label.existing_label_error(job))
            meta_path = label_metadata_path(output)
            saved_meta = json.loads(meta_path.read_text())
            self.assertEqual(saved_meta["attempts"], 3)
            for field in ("relative_id", "episode_index", "length", "fps", "camera", "instruction", "sampling", "source"):
                mismatch = deepcopy(saved_meta)
                mismatch[field] = "different"
                meta_path.write_text(json.dumps(mismatch))
                self.assertIn(field, auto_label.existing_label_error(job))
            meta_path.unlink()
            self.assertIsNotNone(auto_label.existing_label_error(job))
            meta_path.write_text(json.dumps(saved_meta))
            stdout = io.StringIO()
            with patch.object(auto_label, "parse_args", return_value=args), \
                    patch.object(auto_label, "make_jobs", return_value=([job], None, [])), \
                    patch.object(auto_label, "create_client") as create_client, redirect_stdout(stdout):
                auto_label.main()
            create_client.assert_not_called()
            summary = json.loads(stdout.getvalue()[stdout.getvalue().index("{"):])
            self.assertEqual((summary["valid_existing"], summary["annotated"], summary["complete"]), (1, 0, True))

            failed_job = auto_label.AnnotationJob(other, metadata, indices, video_path="mock.mp4")
            with patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, frame)), \
                    patch.object(auto_label, "run_inference", return_value="[{}]") as infer, \
                    redirect_stdout(io.StringIO()):
                result = auto_label.process_job(object(), args, failed_job, None)
            self.assertEqual((result["status"], infer.call_count), ("failed", 3))
            self.assertFalse(other.exists())
            self.assertFalse(label_metadata_path(other).exists())
            self.assertEqual(len(result["errors"]), 3)

            stale_meta = {**saved_meta, "instruction": "A different episode instruction."}
            meta_path.write_text(json.dumps(stale_meta))
            args.api_key = "mock-key"
            stdout = io.StringIO()
            reader = SimpleNamespace(_row_cache={"existing": "cached rows"})
            with patch.object(auto_label, "parse_args", return_value=args), \
                    patch.object(auto_label, "make_jobs", return_value=([job], reader, [])), \
                    patch.object(auto_label, "create_client"), \
                    patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, frame)), \
                    patch.object(auto_label, "process_job", wraps=auto_label.process_job) as process, \
                    patch.object(auto_label, "run_inference", return_value="[{}]") as infer, \
                    redirect_stdout(stdout):
                auto_label.main()
            summary = json.loads(stdout.getvalue()[stdout.getvalue().index("{\n"):])
            self.assertEqual((summary["invalid_labels_before"], summary["failed"], infer.call_count), (1, 1, 3))
            self.assertFalse(summary["complete"])
            self.assertFalse(output.exists())
            self.assertFalse(meta_path.exists())
            worker_reader = process.call_args.args[3]
            self.assertIsNot(worker_reader, reader)
            self.assertEqual(worker_reader._row_cache, {})
            self.assertEqual(reader._row_cache, {"existing": "cached rows"})
            archived_label, archived_meta = map(Path, summary["invalid_existing"][0]["archived"])
            self.assertEqual(json.loads(archived_label.read_text()), labels)
            self.assertEqual(json.loads(archived_meta.read_text()), stale_meta)


if __name__ == "__main__":
    unittest.main()
