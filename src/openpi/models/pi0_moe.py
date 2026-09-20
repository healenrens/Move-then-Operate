import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class RouterMLP(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dense_out = nnx.Linear(hidden_dim, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = nnx.swish(x)
        x = self.dense2(x)
        x = nnx.swish(x)
        return self.dense_out(x)


@dataclasses.dataclass
class _DualExpertContext:
    flow_mse: jax.Array
    route_logits: jax.Array
    predicted_route: jax.Array
    selected_route: jax.Array
    route_labels: jax.Array | None
    route_ce: jax.Array
    expert_mse: jax.Array


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        use_adarms = [False, True, True] if config.pi05 else [False, False, False]
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_expert_keys = ("expert0", "expert1")
        self.action_width = action_expert_config.width
        self.action_in_proj = nnx.Dict(
            **{
                key: nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                for key in self.action_expert_keys
            }
        )
        if config.pi05:
            self.time_mlp_in = nnx.Dict(
                **{
                    key: nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                    for key in self.action_expert_keys
                }
            )
            self.time_mlp_out = nnx.Dict(
                **{
                    key: nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                    for key in self.action_expert_keys
                }
            )
        else:
            self.state_proj = nnx.Dict(
                **{
                    key: nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                    for key in self.action_expert_keys
                }
            )
            self.action_time_mlp_in = nnx.Dict(
                **{
                    key: nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
                    for key in self.action_expert_keys
                }
            )
            self.action_time_mlp_out = nnx.Dict(
                **{
                    key: nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                    for key in self.action_expert_keys
                }
            )
        self.action_out_proj = nnx.Dict(
            **{
                key: nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
                for key in self.action_expert_keys
            }
        )
        self.router = RouterMLP(input_dim=paligemma_config.width, hidden_dim=paligemma_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix_expert(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        expert_index: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        """Embed the suffix for one statically selected action expert."""
        key = self.action_expert_keys[expert_index]
        batch_size = noisy_actions.shape[0]
        time_embedding = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        expert_tokens = []
        expert_masks = []
        expert_ar = []

        action_proj = getattr(self.action_in_proj, key)
        action_tokens = action_proj(noisy_actions)

        if self.pi05:
            time_in = getattr(self.time_mlp_in, key)
            time_out = getattr(self.time_mlp_out, key)
            cond = time_in(time_embedding)
            cond = nnx.swish(cond)
            cond = time_out(cond)
            cond = nnx.swish(cond)
            expert_tokens.append(action_tokens)
            expert_masks.append(jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_))
            expert_ar.extend([True] + [False] * (self.action_horizon - 1))
        else:
            state_proj = getattr(self.state_proj, key)
            state_token = state_proj(obs.state)[:, None, :]
            expert_tokens.append(state_token)
            expert_masks.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            expert_ar.append(True)

            time_tokens = einops.repeat(time_embedding, "b emb -> b s emb", s=self.action_horizon)
            mlp_in = getattr(self.action_time_mlp_in, key)
            mlp_out = getattr(self.action_time_mlp_out, key)
            mixed = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            mixed = mlp_in(mixed)
            mixed = nnx.swish(mixed)
            mixed = mlp_out(mixed)
            mixed = nnx.swish(mixed)
            expert_tokens.append(mixed)
            expert_masks.append(jnp.ones(mixed.shape[:2], dtype=jnp.bool_))
            expert_ar.extend([True] + [False] * (self.action_horizon - 1))
            cond = None

        return (
            jnp.concatenate(expert_tokens, axis=1),
            jnp.concatenate(expert_masks, axis=1),
            jnp.array(expert_ar, dtype=jnp.bool_),
            cond,
        )

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array], list[jax.Array | None]]:
        tokens_per_expert = []
        masks_per_expert = []
        ar_masks_per_expert = []
        adarms_per_expert = []

        for expert_index in range(len(self.action_expert_keys)):
            tokens, mask, ar_mask, adarms = self.embed_suffix_expert(obs, noisy_actions, timestep, expert_index)
            tokens_per_expert.append(tokens)
            masks_per_expert.append(mask)
            ar_masks_per_expert.append(ar_mask)
            adarms_per_expert.append(adarms)

        return tokens_per_expert, masks_per_expert, ar_masks_per_expert, adarms_per_expert

    def _run_prefix(
        self,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        prefix_ar_mask: jax.Array,
        *,
        deterministic: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outputs, _ = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            deterministic=deterministic,
        )
        return outputs[0], prefix_attn_mask, positions

    def _run_expert(
        self,
        prefix_tokens: jax.Array,
        prefix_mask: jax.Array,
        prefix_ar_mask: jax.Array,
        expert_tokens: jax.Array,
        expert_mask: jax.Array,
        expert_ar_mask: jax.Array,
        expert_index: int,
        adarms_cond: jax.Array | None,
        *,
        deterministic: bool,
    ) -> jax.Array:
        input_mask = jnp.concatenate([prefix_mask, expert_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, expert_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        tokens = [prefix_tokens, None, None]
        adarms = [None, None, None]
        tokens[expert_index + 1] = expert_tokens
        adarms[expert_index + 1] = adarms_cond
        outputs, _ = self.PaliGemma.llm(
            tokens,
            mask=attn_mask,
            positions=positions,
            adarms_cond=adarms,
            deterministic=deterministic,
        )
        expert_out = outputs[expert_index + 1]
        return expert_out[:, -self.action_horizon :]

    def _pool_prefix(self, prefix_out: jax.Array, prefix_mask: jax.Array) -> jax.Array:
        mask = prefix_mask.astype(prefix_out.dtype)[..., None]
        summed = jnp.sum(prefix_out * mask, axis=1)
        counts = jnp.sum(mask, axis=1)
        counts = jnp.maximum(counts, 1.0)
        return summed / counts

    def _compute_router_logits(self, prefix_out: jax.Array, prefix_mask: jax.Array) -> jax.Array:
        pooled = self._pool_prefix(prefix_out, prefix_mask)
        return self.router(pooled)

    def _dual_expert_forward(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens_list, suffix_masks_list, suffix_ar_masks_list, adarms_list = self.embed_suffix(observation, x_t, time)

        prefix_out, _, _ = self._run_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            deterministic=not train,
        )
        router_logits = self._compute_router_logits(prefix_out, prefix_mask)
        predicted_route = jnp.argmax(router_logits.astype(jnp.float32), axis=-1)

        route_labels = observation.route_labels
        if train and route_labels is None:
            raise ValueError("route_labels must be provided during training for dual expert routing.")
        if route_labels is not None:
            if route_labels.ndim != 1 or route_labels.shape[0] != router_logits.shape[0]:
                raise ValueError("route_labels must have shape [B] and match batch size")
            route_labels = route_labels.astype(jnp.int32)

        expert_outputs = []
        for idx, key in enumerate(self.action_expert_keys):
            hidden = self._run_expert(
                prefix_tokens,
                prefix_mask,
                prefix_ar_mask,
                suffix_tokens_list[idx],
                suffix_masks_list[idx],
                suffix_ar_masks_list[idx],
                idx,
                adarms_list[idx],
                deterministic=not train,
            )
            expert_outputs.append(getattr(self.action_out_proj, key)(hidden))
        expert_outputs = jnp.stack(expert_outputs, axis=-1)

        if route_labels is not None and train:
            selected_route = route_labels
        else:
            selected_route = predicted_route

        route_onehot = jax.nn.one_hot(
            selected_route,
            num_classes=len(self.action_expert_keys),
            dtype=expert_outputs.dtype,
        )
        v_t = jnp.sum(expert_outputs * route_onehot[:, None, None, :], axis=-1)
        flow_delta = v_t - u_t
        sq_flow = jnp.square(flow_delta)

        time_mask = observation.action_loss_mask
        dim_mask = observation.action_dim_mask
        if time_mask is None:
            time_mask = jnp.ones(sq_flow.shape[:2], dtype=jnp.bool_)
        if dim_mask is None:
            dim_mask = jnp.ones(sq_flow.shape, dtype=jnp.bool_)
        elif dim_mask.ndim == 2:
            dim_mask = jnp.broadcast_to(dim_mask[:, None, :], sq_flow.shape)
        time_mask_f = time_mask.astype(sq_flow.dtype)
        dim_mask_f = dim_mask.astype(sq_flow.dtype)
        combined_mask = dim_mask_f * time_mask_f[..., None]
        masked_sum = jnp.sum(sq_flow * combined_mask, axis=-1)
        # The trainer averages this [B, H] array. Scale its contributions so that
        # the resulting scalar is averaged over valid action elements only.
        loss_scale = masked_sum.size / jnp.sum(combined_mask)
        flow_mse = masked_sum * loss_scale

        expert_sq = jnp.square(expert_outputs - u_t[..., None])
        combined_mask_expert = combined_mask[..., None]
        masked_sum_expert = jnp.sum(expert_sq * combined_mask_expert, axis=-2)
        expert_mse = masked_sum_expert * loss_scale

        if route_labels is not None:
            log_probs = jax.nn.log_softmax(router_logits.astype(jnp.float32), axis=-1)
            route_ce = -jnp.take_along_axis(log_probs, route_labels[:, None], axis=-1)[:, 0]
        else:
            route_ce = jnp.zeros(router_logits.shape[0], dtype=flow_mse.dtype)

        context = _DualExpertContext(
            flow_mse=flow_mse,
            route_logits=router_logits,
            predicted_route=predicted_route,
            selected_route=selected_route,
            route_labels=route_labels,
            route_ce=route_ce,
            expert_mse=expert_mse,
        )
        return context

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        context = self._dual_expert_forward(rng, observation, actions, train=train)
        flow_loss = context.flow_mse
        if context.route_labels is not None:
            flow_loss = flow_loss + context.route_ce[:, None]
        return flow_loss

    def compute_metrics(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        context = self._dual_expert_forward(rng, observation, actions, train=train)

        total_loss = context.flow_mse
        metrics: dict[str, jax.Array] = {}

        if context.route_labels is not None:
            total_loss = total_loss + context.route_ce[:, None]
            metrics["loss_route_ce"] = jnp.mean(context.route_ce)
            metrics["route_accuracy"] = jnp.mean(
                (context.predicted_route == context.route_labels).astype(jnp.float32)
            )
            metrics["route_label_fraction"] = jnp.mean(context.route_labels.astype(jnp.float32))

        metrics["loss_total"] = jnp.mean(total_loss)
        metrics["loss_mse"] = jnp.mean(context.flow_mse)

        expert_mse = jnp.mean(context.expert_mse, axis=(0, 1))
        for idx, value in enumerate(expert_mse):
            metrics[f"loss_mse_expert{idx}"] = value

        metrics["route_pred_fraction"] = jnp.mean(context.predicted_route.astype(jnp.float32))
        metrics["route_selected_fraction"] = jnp.mean(context.selected_route.astype(jnp.float32))

        return metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_out, _, _ = self._run_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            deterministic=True,
        )
        router_logits = self._compute_router_logits(prefix_out, prefix_mask)
        route_indices = jnp.argmax(router_logits.astype(jnp.float32), axis=-1)

        def step(carry):
            x_t, time = carry
            suffix_tokens_list, suffix_masks_list, suffix_ar_masks_list, adarms_list = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            v_t_per_expert = []
            for idx, key in enumerate(self.action_expert_keys):
                hidden = self._run_expert(
                    prefix_tokens,
                    prefix_mask,
                    prefix_ar_mask,
                    suffix_tokens_list[idx],
                    suffix_masks_list[idx],
                    suffix_ar_masks_list[idx],
                    idx,
                    adarms_list[idx],
                    deterministic=True,
                )
                v_t_per_expert.append(getattr(self.action_out_proj, key)(hidden))
            v_t_stack = jnp.stack(v_t_per_expert, axis=-1)
            route_weights = jax.nn.one_hot(
                route_indices,
                len(self.action_expert_keys),
                dtype=v_t_stack.dtype,
            )
            v_t = jnp.sum(v_t_stack * route_weights[:, None, None, :], axis=-1)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
