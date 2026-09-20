"""CPU save/restore/continue regression using the production checkpoint format."""
import dataclasses
from pathlib import Path
import tempfile
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import checkpoints
from openpi.training.utils import TrainState


class TinyModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array([[0.2], [-0.3]], dtype=jnp.float32), role="action_projection")
        self.bias = nnx.Param(jnp.array([0.1], dtype=jnp.float32), role="action_bias")

    def __call__(self, x):
        return x @ self.weight.value + self.bias.value


def make_state(ema_decay):
    model_def, params = nnx.split(TinyModel())
    tx = optax.adam(0.01)
    return TrainState(step=jnp.array(0), params=params, model_def=model_def, tx=tx,
                      opt_state=tx.init(params), ema_decay=ema_decay,
                      ema_params=None if ema_decay is None else jax.tree.map(jnp.copy, params))


def update(state):
    def loss(params):
        model = nnx.merge(state.model_def, params)
        prediction = model(jnp.array([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32))
        return jnp.mean((prediction - jnp.array([[0.4], [-0.2]], dtype=jnp.float32)) ** 2)

    gradients = jax.grad(loss)(state.params)
    updates, opt_state = state.tx.update(gradients, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema = None if state.ema_params is None else jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, params)
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state, ema_params=ema)


class CheckpointResumeIntegrationTest(unittest.TestCase):
    def test_save_restore_and_next_update_with_and_without_ema(self):
        with tempfile.TemporaryDirectory(prefix="mto-resume-") as directory:
            for ema_decay in (None, 0.9):
                with self.subTest(ema_decay=ema_decay):
                    path = Path(directory) / str(ema_decay)
                    before = update(update(make_state(ema_decay)))
                    manager, resuming = checkpoints.initialize_checkpoint_dir(
                        path, keep_period=None, overwrite=False, resume=False)
                    self.assertFalse(resuming)
                    checkpoints.save_state(manager, before, 2)
                    manager.wait_until_finished()
                    manager.close()

                    manager, resuming = checkpoints.initialize_checkpoint_dir(
                        path, keep_period=None, overwrite=False, resume=True)
                    self.assertTrue(resuming)
                    # Production resume starts from jax.eval_shape, not initialized arrays.
                    template = jax.eval_shape(lambda: make_state(ema_decay))
                    restored = checkpoints.restore_state(manager, template, step=2)
                    manager.close()
                    self.assertEqual(int(restored.step), 2)
                    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(before), strict=True):
                        np.testing.assert_array_equal(actual, expected)
                    for field in ("params", "ema_params"):
                        actual, reference = getattr(restored, field), getattr(template, field)
                        if actual is None:
                            self.assertIsNone(reference)
                            continue
                        for name in ("weight", "bias"):
                            self.assertIsInstance(actual[name], nnx.VariableState)
                            self.assertIs(actual[name].type, nnx.Param)
                            self.assertEqual(actual[name].get_metadata(), reference[name].get_metadata())
                            self.assertIsInstance(reference[name].value, jax.ShapeDtypeStruct)

                    # Restored Adam moments, raw parameters and EMA all participate.
                    continued, uninterrupted = update(restored), update(before)
                    self.assertEqual(int(continued.step), 3)
                    for actual, expected in zip(jax.tree.leaves(continued), jax.tree.leaves(uninterrupted), strict=True):
                        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
