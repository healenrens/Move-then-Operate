"""Publication configuration, redacted annotation errors, and failed exit status."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
import runpy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError
from PIL import Image

from mto import auto_label


class ReleaseConfigurationTest(unittest.TestCase):
    def test_environment_config_and_failed_jobs(self):
        env = {"ARK_MODEL_ID": "fixture-model", "ARK_BASE_URL": "http://localhost"}
        with patch.dict(os.environ, env, clear=True), patch("sys.argv", ["annotate", "--root_dir", "/data/fixture"]):
            args = auto_label.parse_args()
        self.assertEqual((args.model_id, args.base_url), ("fixture-model", "http://localhost"))
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["annotate", "--root_dir", "/data/fixture"]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_result:
            auto_label.parse_args()
        self.assertEqual(exit_result.exception.code, 2)
        with patch.object(auto_label, "parse_args", return_value=args), \
                patch.object(auto_label, "make_jobs", return_value=([], None, [])), \
                patch.object(auto_label, "create_client") as client, redirect_stdout(io.StringIO()):
            self.assertEqual(auto_label.main(), 1)
        client.assert_not_called()
        args.max_attempts = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode0.json"
            metadata = auto_label.build_metadata(args, "fixture/clean", 0, 2, 30, "Move the object.", {})
            job = auto_label.AnnotationJob(path, metadata, [0, 1], video_path="fixture.mp4")
            request = httpx.Request("POST", "http://localhost/responses")
            # Construct values at runtime so no literal credential is part of the release.
            key = "fixture" + "-client-value"
            env_key = "fixture" + "-environment-value"
            failure = APIStatusError(f"provider echoed {key} and {env_key}", response=httpx.Response(500, request=request), body=None)
            stdout = io.StringIO()
            with patch.dict(os.environ, {"ARK_API_KEY": env_key}), \
                    patch.object(auto_label, "read_frame_at", side_effect=lambda _, index: (index, Image.new("RGB", (8, 8)))), \
                    patch.object(auto_label, "run_inference", side_effect=failure), redirect_stdout(stdout):
                result = auto_label.process_job(SimpleNamespace(api_key=key), args, job, None)
            captured = stdout.getvalue() + json.dumps(result)
            captured += Path(auto_label.sidecar_path(str(path), "_phases_error.txt")).read_text()
            self.assertNotIn(key, captured)
            self.assertNotIn(env_key, captured)
            self.assertIn("[REDACTED]", captured)
            self.assertFalse(path.exists())

            checker = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_release.py"))
            publication = Path(directory) / "publication"
            publication.mkdir()
            samples = {"private.txt": "/" + "Users/fixture/project",
                       "endpoint.txt": "ep-" + "20260101000000-fixture",
                       "key.txt": "sk-" + "x" * 24,
                       "language.txt": chr(0x4e2d),
                       "private_key.txt": "-----BEGIN " + "PRIVATE KEY-----"}
            for name, value in samples.items():
                (publication / name).write_text(value)
            findings = checker["check_files"](publication, samples)
            self.assertEqual(len(findings), 5)
            self.assertEqual({finding[2] for finding in findings},
                {"private path", "personal model endpoint", "credential candidate", "non-English text", "private key"})
            output = io.StringIO()
            with patch("sys.argv", ["check_release", "--root", str(publication)]), redirect_stdout(output):
                self.assertEqual(checker["main"](), 1)
            for value in samples.values():
                self.assertNotIn(value, output.getvalue())


if __name__ == "__main__":
    unittest.main()
