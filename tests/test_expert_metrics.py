"""Ground-truth expert metrics retain valid-element weighting and legacy keys."""
from types import SimpleNamespace
import unittest

import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_moe import Pi0, _DualExpertContext, _routed_expert_metrics


class ExpertMetricsTest(unittest.TestCase):
    def test_group_weighting_missing_experts_and_empty_masks(self):
        for labels, counts in (([0, 1, 1], [32, 64, 32]), ([0, 0, 0], [32, 64, 32]),
                               ([0, 1, 1], [0, 0, 0])):
            counts = jnp.array(counts, dtype=jnp.float32)
            raw_sse = jnp.array([[32., 96.], [128., 256.], [160., 192.]]) * (counts > 0)[:, None]
            scale = 6 / max(float(counts.sum()), 1.)
            expert = jnp.stack([raw_sse * scale, jnp.zeros_like(raw_sse)], axis=1)
            context = _DualExpertContext(flow_mse=expert[..., 0], route_logits=jnp.zeros((3, 2)),
                predicted_route=jnp.array([1, 0, 0]), selected_route=jnp.array(labels),
                route_labels=jnp.array(labels), route_ce=jnp.zeros(3), expert_mse=expert,
                valid_elements_per_sample=counts)
            result = _routed_expert_metrics(context)
            for index in (0, 1):
                group = np.array(labels) == index
                denominator = float(counts[group].sum())
                expected = float(raw_sse[group, index].sum()) / max(denominator, 1.)
                self.assertEqual(float(result[f"valid_elements_expert{index}"]), denominator)
                self.assertAlmostEqual(float(result[f"loss_mse_expert{index}_masked"]), expected, places=6)
            model = SimpleNamespace(_dual_expert_forward=lambda *args, **kwargs: context)
            metrics = Pi0.compute_metrics(model, None, None, None, train=True)
            for index in (0, 1):
                self.assertAlmostEqual(float(metrics[f"loss_mse_expert{index}"]),
                                      float(expert[..., index].mean()), places=6)
            self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
