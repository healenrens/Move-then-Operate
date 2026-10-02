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
from openai import OpenAI
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
                                  max_new_tokens=2048, max_attempts=3, concurrency=1, resume=True, api_key=None,
                                  gripper_event_threshold=0.02, event_context_frames=2, mosaic_tile_size=8)
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
            requests = []

            def responses_endpoint(request):
                self.assertEqual(request.method, "POST")
                self.assertEqual(request.url.path, "/responses")
                body = json.loads(request.content)
                requests.append(body)
                self.assertEqual(body["model"], args.model_id)
                self.assertEqual(body["max_output_tokens"], args.max_new_tokens)
                self.assertNotIn("messages", body)
                self.assertNotIn("max_tokens", body)
                for message in body["input"]:
                    self.assertTrue(all(part["type"] in ("input_text", "input_image")
                                        for part in message["content"]))
                user_content = body["input"][-1]["content"]
                images = [part for part in user_content if part["type"] == "input_image"]
                self.assertEqual(len(images), len(indices))
                self.assertTrue(all(part["image_url"].startswith("data:image/jpeg;base64,") for part in images))
                indexed_text = [part["text"] for part in user_content
                                if part["type"] == "input_text" and part["text"].startswith("frame_idx=")]
                self.assertEqual(indexed_text, [f"frame_idx={index}" for index in indices])
                if len(requests) == 1:
                    return httpx.Response(429, json={"error": {"message": "fixture rate limit", "type": "rate_limit"}})
                if len(requests) <= 3:
                    self.assertIn("Previous attempt issues", user_content[-1]["text"])
                else:
                    self.assertIn("Boundary calibration stage", user_content[-1]["text"])
                text = "[invalid JSON]" if len(requests) == 2 else json.dumps(labels)
                midpoint = len(text) // 2
                return httpx.Response(200, json={
                    "id": "resp_fixture", "object": "response", "created_at": 0,
                    "model": args.model_id, "status": "completed", "output": [
                        {"type": "reasoning", "id": "reasoning_fixture", "summary": [
                            {"type": "summary_text", "text": "This reasoning is not label JSON."}]},
                        {"type": "message", "id": "message_fixture", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": fragment, "annotations": []}
                                     for fragment in (text[:midpoint], text[midpoint:])]},
                    ],
                })

            client = OpenAI(api_key="mock-key", base_url=args.base_url, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(responses_endpoint)))
            with patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, frame)), \
                    client, redirect_stdout(io.StringIO()):
                result = auto_label.process_job(client, args, job, None)
            self.assertEqual((result["status"], len(requests)), ("annotated", 4))
            self.assertEqual(load_phase_labels(output, 720), (labels, None))
            self.assertIsNone(auto_label.existing_label_error(job))
            meta_path = label_metadata_path(output)
            saved_meta = json.loads(meta_path.read_text())
            self.assertEqual(saved_meta["attempts"], 4)
            self.assertEqual(saved_meta["model"]["api"], "responses")
            for field in ("relative_id", "episode_index", "length", "fps", "camera", "instruction", "sampling", "source", "model", "rule_version", "visual_layout", "temporal_evidence"):
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
                self.assertEqual(auto_label.main(), 0)
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

            corrected = deepcopy(labels)
            corrected[0]["phases"][0]["end_frame_idx"] = 89
            corrected[0]["phases"][1]["start_frame_idx"] = 90
            self.assertEqual(auto_label.calibration_error(labels, corrected), "")
            semantic_change = deepcopy(corrected)
            semantic_change[0]["primary_arm"] = "right"
            self.assertTrue(auto_label.calibration_error(labels, semantic_change))
            with patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, frame)), \
                    patch.object(auto_label, "run_inference", side_effect=[json.dumps(labels)] + [json.dumps(semantic_change)] * 3), \
                    redirect_stdout(io.StringIO()):
                result = auto_label.process_job(object(), args, failed_job, None)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(other.exists())

            import numpy as np
            state = np.zeros((8, 14), dtype=np.float32)
            state[2:5, 6] = [0.2, 0.4, 0.6]
            state[5:, 6] = 0.6
            state[5:, 13] = 1.0
            events = auto_label.extract_gripper_events(state, 30, 0.02)
            self.assertEqual([(event["arm"], event["start_frame_idx"], event["end_frame_idx"])
                              for event in events], [("left", 1, 4), ("right", 4, 5)])
            selected = auto_label.sample_event_frames(8, 30, 5, 6, events, 0)
            self.assertTrue({0, 1, 4, 5, 7}.issubset(selected))
            dense = [{"start_frame_idx": i, "end_frame_idx": i + 1} for i in range(719)]
            selected = auto_label.sample_event_frames(720, 30, 5, 16, dense, 2)
            self.assertEqual((selected[0], selected[-1], len(selected)), (0, 719, 16))
            colors = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
            mosaic = auto_label.make_camera_mosaic(colors, auto_label.CAMERA_KEYS, 8)
            self.assertEqual(mosaic.size, (24, 32))
            self.assertEqual([mosaic.getpixel((i * 8 + 4, 28)) for i in range(3)],
                             [(255, 0, 0), (0, 128, 0), (0, 0, 255)])
            episode_job = auto_label.AnnotationJob(other, deepcopy(metadata), indices, episode=object())
            episode_job.metadata["visual_layout"]["cameras"] = list(auto_label.CAMERA_KEYS)
            reader_views = SimpleNamespace(read_rgb=lambda ep, camera, frames: [colors[auto_label.CAMERA_KEYS.index(camera)]] * len(frames))
            with patch.object(auto_label, "run_inference", side_effect=[json.dumps(labels), json.dumps(corrected)]) as infer, \
                    redirect_stdout(io.StringIO()):
                result = auto_label.process_job(object(), args, episode_job, reader_views)
            self.assertEqual(result["status"], "annotated")
            self.assertEqual(infer.call_count, 2)
            self.assertEqual(json.loads(other.read_text()), corrected)
            other.unlink()
            label_metadata_path(other).unlink()

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
                self.assertEqual(auto_label.main(), 1)
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
