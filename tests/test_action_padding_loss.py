"""Exercise production flow matching with controlled expert velocity predictions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as observation_model
from openpi.models.pi0_moe import Pi0, _routed_expert_metrics


def controlled_context(predictions, observation, actions, rng):
    batch = len(actions)
    prefix = jnp.zeros((batch, 1, 1))
    mask = jnp.ones((batch, 1), dtype=bool)
    stub = SimpleNamespace(
        action_expert_keys=("expert0", "expert1"),
        action_out_proj=SimpleNamespace(expert0=lambda x: x, expert1=lambda x: x),
        embed_prefix=lambda obs: (prefix, mask, jnp.array([False])),
        embed_suffix=lambda obs, x, t: ([prefix, prefix], [mask, mask], [jnp.array([True])] * 2, [None] * 2),
        _run_prefix=lambda *args, **kwargs: (prefix, None, None),
        _compute_router_logits=lambda *args: jnp.tile(jnp.array([[1., 0.]]), (batch, 1)),
        _run_expert=lambda *args, **kwargs: predictions[..., args[6]],
    )
    return Pi0._dual_expert_forward(stub, rng, observation, actions, train=True)


class ActionPaddingLossTest(unittest.TestCase):
    def test_loss_masks_noise_targets_and_jit_gradients(self):
        rng = jax.random.key(11)
        actions = jnp.zeros((2, 3, 32)).at[:, :, :14].set(0.25)
        observation = SimpleNamespace(route_labels=jnp.array([0, 1]),
                                      action_loss_mask=jnp.array([[True, False, False], [True, True, False]]),
                                      action_dim_mask=jnp.ones((2, 32), dtype=bool))
        predictions = jnp.zeros((2, 3, 32, 2))
        with patch.object(observation_model, "preprocess_observation", side_effect=lambda rng, obs, train: obs):
            context = controlled_context(predictions, observation, actions, rng)
            noise = jax.random.normal(jax.random.split(rng, 3)[1], actions.shape)
            valid = np.broadcast_to(np.asarray(observation.action_loss_mask)[..., None], actions.shape)
            expected = np.asarray((noise - actions) ** 2)
            self.assertAlmostEqual(float(context.flow_mse.mean()), float(expected[valid].mean()), places=6)
            np.testing.assert_array_equal(context.valid_elements_per_sample, [32, 64])
            # The padded zero action target still requires matching the sampled noise.
            self.assertGreater(float(jnp.sum(expected[:, :, 14:] * valid[:, :, 14:])), 0)
            loss = lambda pred: controlled_context(pred, observation, actions, rng).flow_mse.mean()
            gradients = jax.jit(jax.grad(loss))(predictions)
            for row, route in enumerate((0, 1)):
                np.testing.assert_array_equal(gradients[row, ..., 1 - route], 0)
                np.testing.assert_array_equal(gradients[row, ~observation.action_loss_mask[row], :, route], 0)
                self.assertGreater(float(jnp.linalg.norm(gradients[row, observation.action_loss_mask[row], 14:, route])), 0)
            metrics = _routed_expert_metrics(context)
            for route in (0, 1):
                self.assertAlmostEqual(float(metrics[f"loss_mse_expert{route}_masked"]),
                                       float(expected[route][valid[route]].mean()), places=6)


if __name__ == "__main__":
    unittest.main()
