"""Automatic checkpoint selection and W&B identity before the first save."""
from dataclasses import make_dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import jax
import numpy as np

from mto.train_utils import init_wandb
from openpi.training import checkpoints
from test_checkpoint_resume import make_state, update


class AutoResumeTest(unittest.TestCase):
    def test_latest_complete_checkpoint_and_wandb_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "experiment"
            manager, resuming = checkpoints.initialize_checkpoint_dir(path, keep_period=1, overwrite=False, resume=True)
            self.assertFalse(resuming)
            manager.close()
            wandb = MagicMock()
            wandb.run.id = "fixture-run"
            config_type = make_dataclass("TrackingConfig", ["checkpoint_dir", "project_name", "resume", "exp_name"])
            config = config_type(path, "fixture-project", True, "fixture-experiment")
            with patch.dict("sys.modules", {"wandb": wandb}):
                init_wandb(config, resuming=False, enabled=True)
            self.assertEqual((path / "wandb_id.txt").read_text(), "fixture-run")
            wandb.reset_mock()
            (path / "wandb_id.txt").write_text("fixture-run")
            with patch.dict("sys.modules", {"wandb": wandb}):
                init_wandb(config, resuming=False, enabled=True)
            wandb.init.assert_called_once_with(id="fixture-run", resume="must", project="fixture-project")

            manager, resuming = checkpoints.initialize_checkpoint_dir(path, keep_period=1, overwrite=False, resume=True)
            self.assertFalse(resuming)
            state = update(make_state(0.9))
            checkpoints.save_state(manager, state, 1)
            manager.wait_until_finished()
            state = update(state)
            checkpoints.save_state(manager, state, 2)
            manager.wait_until_finished()
            manager.close()
            (path / "3.orbax-checkpoint-tmp").mkdir()
            manager, resuming = checkpoints.initialize_checkpoint_dir(path, keep_period=1, overwrite=False, resume=True)
            self.assertTrue(resuming)
            self.assertEqual(manager.latest_step(), 2)
            restored = checkpoints.restore_state(manager, jax.eval_shape(lambda: make_state(0.9)))
            manager.close()
            self.assertEqual(int(restored.step), 2)
            for actual, expected in zip(jax.tree.leaves(update(restored)), jax.tree.leaves(update(state)), strict=True):
                np.testing.assert_array_equal(actual, expected)
            wandb.reset_mock()
            with patch.dict("sys.modules", {"wandb": wandb}):
                init_wandb(config, resuming=True, enabled=True)
            wandb.init.assert_called_once_with(id="fixture-run", resume="must", project="fixture-project")


if __name__ == "__main__":
    unittest.main()
