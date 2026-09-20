import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_moe import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # If true, freeze the Paligemma backbone (language + vision encoders) and only train routing / experts.
    freeze_paligemma: bool = False

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_moe import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        gemma_params_filter = nnx_utils.PathRegex(".*PaliGemma/llm/.*")
        # Expert suffixes belong to modules inside the shared llm tree.
        action_expert_params_filter = nnx_utils.PathRegex(
            r".*PaliGemma/llm/(?:layers/)?(?:attn/)?[^/]+_[1-9][0-9]*(?:/.*)?"
        )
        base_llm_filter = nnx.All(gemma_params_filter, nnx.Not(action_expert_params_filter))
        img_params_filter = nnx_utils.PathRegex(".*PaliGemma/img.*")
        non_lora_filter = nnx.Not(nnx_utils.PathRegex(".*lora.*"))
        freeze_nodes: list[nnx.filterlib.Filter] = []
        if "lora" in self.paligemma_variant:
            freeze_nodes.append(nnx.All(base_llm_filter, non_lora_filter))
            freeze_nodes.append(img_params_filter)
        if "lora" in self.action_expert_variant:
            freeze_nodes.append(nnx.All(action_expert_params_filter, non_lora_filter))
        if self.freeze_paligemma:
            freeze_nodes.append(base_llm_filter)
            freeze_nodes.append(img_params_filter)
        if not freeze_nodes:
            return nnx.Nothing
        if len(freeze_nodes) == 1:
            return freeze_nodes[0]
        return nnx.Any(*freeze_nodes)
