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

import json
import pathlib
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


_CAMERAS = {
    "head_camera": "base_0_rgb",
    "left_camera": "left_wrist_0_rgb",
    "right_camera": "right_wrist_0_rgb",
}
_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
_NORMALIZATION_EPS = 1e-6


class QposNormalizer:
    """Read the standalone action/proprio statistics used by the phase dataset."""

    def __init__(self, path: str | pathlib.Path, method: Literal["zscore", "min_max"] = "zscore"):
        with pathlib.Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        statistics = payload.get("statistics", payload)
        self.method = method
        self.offset = {}
        self.scale = {}
        for key in ("action", "proprio"):
            dims = [statistics[key][f"dim_{i}"] for i in _JOINT_INDICES]
            if method == "zscore":
                self.offset[key] = np.array([dim["mean"] for dim in dims], dtype=np.float32)
                self.scale[key] = np.array([dim["std"] for dim in dims], dtype=np.float32) + _NORMALIZATION_EPS
            else:
                low = np.array([dim["percentile_1"] for dim in dims], dtype=np.float32)
                high = np.array([dim["percentile_99"] for dim in dims], dtype=np.float32)
                self.offset[key] = low
                self.scale[key] = np.maximum(high - low, _NORMALIZATION_EPS)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        output = np.array(state, dtype=np.float32, copy=True)
        joints = (output[..., _JOINT_INDICES] - self.offset["proprio"]) / self.scale["proprio"]
        if self.method == "min_max":
            joints = np.clip(joints, 0.0, 1.0)
        output[..., _JOINT_INDICES] = joints
        return output

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        output = np.array(actions, dtype=np.float32, copy=True)
        output[..., _JOINT_INDICES] = (
            output[..., _JOINT_INDICES] * self.scale["action"] + self.offset["action"]
        )
        return output


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
    To use shared normalization, pass the same JSON for both phase paths.
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
        self._rng = jax.random.key(seed)

    @property
    def metadata(self) -> dict:
        return {
            "model": "move_then_operate",
            "phases": ["move", "operate"],
            "action_horizon": self._action_horizon,
            "action_dim": 14,
            "action_type": "absolute_qpos",
        }

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
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
