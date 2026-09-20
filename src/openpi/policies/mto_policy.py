"""Single-observation MTO inference with one action expert per generated chunk.

Inputs are RGB uint8 HWC images under ``images`` (head_camera, left_camera,
right_camera), a raw 14-D ``state`` and a task ``prompt``. Outputs contain a
full chunk of absolute joint/gripper commands under ``actions``. The caller
executes its configured number of commands, observes again, and requests the
next chunk; routing never changes within a flow integration.

Weights, the SentencePiece model and move/operate normalization JSON files are
loaded from independent local paths. This path implements the qpos training
contract, with twelve joint deltas relative to the chunk's initial state and
two absolute gripper values. It does not use the Aloha coordinate conversion.
"""

from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy
from PIL import Image

from openpi.models import model as model_lib
from openpi.models import mto_config
from openpi.models import pi0_moe
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils
from mto.normalization import ACTION_REPRESENTATION, QposNormalizer


_CAMERAS = {
    "head_camera": "base_0_rgb",
    "left_camera": "left_wrist_0_rgb",
    "right_camera": "right_wrist_0_rgb",
}
_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


class _CachedExpertDecoder(nnx.Module):
    """Keep the loaded model unchanged and compile a decoder for each expert."""

    def __init__(self, model: pi0_moe.Pi0):
        self.model = model

    def encode_context(self, observation: model_lib.Observation):
        observation = model_lib.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = self.model.embed_prefix(observation)
        attention_mask = pi0_moe.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        outputs, kv_cache = self.model.PaliGemma.llm(
            [tokens, None, None],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            deterministic=True,
        )
        logits = self.model._compute_router_logits(outputs[0], mask)
        return mask, kv_cache, jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    def decode_expert(
        self,
        prefix_mask,
        kv_cache,
        state,
        noise,
        *,
        expert_index: int,
        num_steps: int,
    ):
        # expert_index is a static JIT argument. The other expert's projections,
        # attention and MLPs are never called or traced into this decoder.
        observation = model_lib.Observation(images={}, image_masks={}, state=state)
        key = self.model.action_expert_keys[expert_index]
        batch_size = state.shape[0]
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            tokens, mask, ar_mask, adarms = self.model.embed_suffix_expert(
                observation, x_t, jnp.broadcast_to(time, (batch_size,)), expert_index
            )
            suffix_attention = pi0_moe.make_attn_mask(mask, ar_mask)
            prefix_attention = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, tokens.shape[1], prefix_mask.shape[1])
            )
            attention_mask = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
            slots = [None, None, None]
            conditions = [None, None, None]
            slots[expert_index + 1] = tokens
            conditions[expert_index + 1] = adarms
            outputs, _ = self.model.PaliGemma.llm(
                slots,
                mask=attention_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=conditions,
                deterministic=True,
            )
            hidden = outputs[expert_index + 1][:, -self.model.action_horizon :]
            velocity = getattr(self.model.action_out_proj, key)(hidden)
            return x_t + dt * velocity, time + dt

        def unfinished(carry):
            return carry[1] >= -dt / 2

        actions, _ = jax.lax.while_loop(unfinished, step, (noise, 1.0))
        return actions


class MtoPolicy(base_policy.BasePolicy):
    """A single-robot policy. Each request selects and runs exactly one expert.

    ``params_path`` points directly to an OpenPI ``params`` directory. The
    config must describe the saved model, including LoRA variants and horizon.
    Move and operate each load their own independently computed statistics.
    """

    def __init__(
        self,
        *,
        params_path: str,
        move_norm_stats_path: str,
        operate_norm_stats_path: str,
        tokenizer_path: str,
        config: mto_config.DualExpertPi0Config,
        normalize_method: Literal["zscore", "min_max"] = "zscore",
        num_steps: int = 10,
        seed: int = 0,
    ):
        params = model_lib.restore_params(params_path, dtype=jnp.dtype(config.dtype))
        model = config.load(params, remove_extra_params=False)
        model.eval()
        decoder = _CachedExpertDecoder(model)
        self._encode_context = nnx_utils.module_jit(decoder.encode_context)
        self._decode_expert = nnx_utils.module_jit(
            decoder.decode_expert, static_argnames=("expert_index", "num_steps")
        )
        self._normalizers = (
            QposNormalizer(move_norm_stats_path, normalize_method),
            QposNormalizer(operate_norm_stats_path, normalize_method),
        )
        self._tokenizer = PaligemmaTokenizer(max_len=config.max_token_len, model_path=tokenizer_path)
        self._action_dim = config.action_dim
        self._action_horizon = config.action_horizon
        self._num_steps = num_steps
        self._seed = seed
        self._rng = jax.random.key(seed)

    @property
    def metadata(self) -> dict:
        return {
            "model": "move_then_operate",
            "phases": ["move", "operate"],
            "action_horizon": self._action_horizon,
            "action_dim": 14,
            "action_type": "absolute_qpos",
            "normalization_mode": "per-expert",
            "normalization_phases": [n.metadata.get("phase") for n in self._normalizers],
            "action_representation": ACTION_REPRESENTATION,
            "state_layout": "left_arm6,left_gripper,right_arm6,right_gripper",
        }

    def reset(self, seed: int | None = None) -> None:
        self._rng = jax.random.key(self._seed if seed is None else seed)

    def _observation(self, obs: dict):
        base_state = np.asarray(obs["state"], dtype=np.float32)
        images = {}
        for source, destination in _CAMERAS.items():
            pil_image = Image.fromarray(np.asarray(obs["images"][source]))
            resized = pil_image.resize((224, 224), resample=Image.Resampling.BICUBIC)
            image = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
            images[destination] = jnp.asarray(image[None, ...])
        prompt = f"{obs['prompt'].strip().replace('_', ' ')}\n"
        tokens, prompt_mask = self._tokenizer.tokenize(prompt)
        observation = model_lib.Observation(
            images=images,
            image_masks={name: jnp.ones((1,), dtype=bool) for name in images},
            state=jnp.asarray(np.pad(base_state, (0, self._action_dim - 14))[None, ...]),
            tokenized_prompt=jnp.asarray(tokens[None, ...], dtype=jnp.int32),
            tokenized_prompt_mask=jnp.asarray(prompt_mask[None, ...]),
        )
        return base_state, observation

    def warmup(self) -> None:
        """Compile both static decoders before the service becomes ready."""
        base, observation = self._observation({
            "state": np.zeros(14, np.float32), "prompt": "warmup",
            "images": {name: np.zeros((224, 224, 3), np.uint8) for name in _CAMERAS},
        })
        prefix_mask, cache, _ = self._encode_context(observation)
        noise = jnp.zeros((1, self._action_horizon, self._action_dim), jnp.float32)
        for expert_index, normalizer in enumerate(self._normalizers):
            state = np.pad(normalizer.normalize_state(base), (0, self._action_dim - 14))[None]
            sampled = self._decode_expert(prefix_mask, cache, jnp.asarray(state), noise,
                                          expert_index=expert_index, num_steps=self._num_steps)
            jax.block_until_ready(sampled)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        base_state, observation = self._observation(obs)
        prefix_mask, kv_cache, probabilities = self._encode_context(observation)
        probabilities = np.asarray(probabilities[0])
        expert_index = int(np.argmax(probabilities))
        normalizer = self._normalizers[expert_index]
        normalized_state = normalizer.normalize_state(base_state)
        state = jnp.asarray(np.pad(normalized_state, (0, self._action_dim - 14))[None, ...])

        self._rng, noise_rng = jax.random.split(self._rng)
        if noise is None:
            initial_noise = jax.random.normal(noise_rng, (1, self._action_horizon, self._action_dim))
        else:
            initial_noise = jnp.asarray(noise, dtype=jnp.float32)[None, ...]
        sampled = self._decode_expert(
            prefix_mask,
            kv_cache,
            state,
            initial_noise,
            expert_index=expert_index,
            num_steps=self._num_steps,
        )
        delta_actions = normalizer.unnormalize_actions(np.asarray(sampled[0, :, :14]))
        absolute_actions = delta_actions.copy()
        absolute_actions[:, _JOINT_INDICES] += base_state[_JOINT_INDICES]
        return {
            "actions": absolute_actions,
            "actions_delta": delta_actions,
            "route_index": expert_index,
            "phase": ("move", "operate")[expert_index],
            "route_probabilities": probabilities,
        }
