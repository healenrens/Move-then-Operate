import dataclasses
import logging
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _tree_dot(x, y):
    leaves_x, tree_x = jax.tree_util.tree_flatten(x)
    leaves_y, tree_y = jax.tree_util.tree_flatten(y)

    if tree_x != tree_y:
        raise ValueError("Tree structures do not match for dot product")

    total = jnp.zeros((), dtype=jnp.float32)
    for lx, ly in zip(leaves_x, leaves_y):
        if lx is None or ly is None:
            continue
        lx_arr = jnp.asarray(lx)
        ly_arr = jnp.asarray(ly)
        total = total + jnp.vdot(lx_arr.astype(jnp.float32), ly_arr.astype(jnp.float32))
    return total


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = False):
    if not enabled:
        return

    import wandb

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    run_id_path = ckpt_dir / "wandb_id.txt"
    if resuming and run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        run_id_path.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    flat_ref = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    def _get_dtype(value):
        if isinstance(value, jax.ShapeDtypeStruct):
            return value.dtype
        if hasattr(value, "dtype"):
            return value.dtype
        return None

    for key, ref_value in flat_ref.items():
        if key not in flat_loaded:
            continue
        ref_dtype = _get_dtype(ref_value)
        if ref_dtype is None:
            continue
        loaded_value = flat_loaded[key]
        if hasattr(loaded_value, "dtype") and loaded_value.dtype != ref_dtype:
            flat_loaded[key] = loaded_value.astype(ref_dtype)

    loaded_params = traverse_util.unflatten_dict(flat_loaded)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    diag_metrics: dict[str, jax.Array] = {}

    if config.log_gradient_diagnostics:
        epsilon = jnp.array(1e-8, dtype=jnp.float32)

        if not hasattr(model, "_dual_expert_forward"):
            raise AttributeError(
                "log_gradient_diagnostics requires a model providing `_dual_expert_forward` (e.g., Pi0 MoE)."
            )

        base_vlm_filter = nnx.All(
            config.trainable_filter,
            nnx_utils.PathRegex(r".*PaliGemma/llm/.*"),
            nnx.Not(nnx_utils.PathRegex(r".*PaliGemma/llm/(?:layers/)?(?:attn/)?[^/]+_[1-9][0-9]*(?:/.*)?")),
        )
        grads_vlm = grads.filter(base_vlm_filter)
        grad_norm_vlm = optax.global_norm(grads_vlm)

        expert0_filter = nnx.All(config.trainable_filter, nnx_utils.PathRegex(".*expert0.*"))
        expert1_filter = nnx.All(config.trainable_filter, nnx_utils.PathRegex(".*expert1.*"))

        grads_expert0 = grads.filter(expert0_filter)
        grads_expert1 = grads.filter(expert1_filter)

        diag_metrics["grad_norm_vlm"] = grad_norm_vlm
        diag_metrics["grad_norm_expert0_heads"] = optax.global_norm(grads_expert0)
        diag_metrics["grad_norm_expert1_heads"] = optax.global_norm(grads_expert1)

        vlm_diff_state = nnx.DiffState(0, base_vlm_filter)
        weights_expert0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
        weights_expert1 = jnp.array([0.0, 1.0], dtype=jnp.float32)

        diag_rng, context_rng = jax.random.split(train_rng)

        def expert_component_loss(
            mdl: _model.BaseModel,
            rng_key: at.KeyArrayLike,
            obs_arg: _model.Observation,
            act_arg: _model.Actions,
            weights: jax.Array,
        ) -> jax.Array:
            context = mdl._dual_expert_forward(rng_key, obs_arg, act_arg, train=True)
            per_expert = jnp.mean(context.expert_mse, axis=(0, 1))
            return jnp.dot(per_expert.astype(jnp.float32), weights.astype(jnp.float32))

        _, grad_vlm_e0 = nnx.value_and_grad(expert_component_loss, argnums=vlm_diff_state)(
            model, diag_rng, observation, actions, weights_expert0
        )
        _, grad_vlm_e1 = nnx.value_and_grad(expert_component_loss, argnums=vlm_diff_state)(
            model, diag_rng, observation, actions, weights_expert1
        )

        norm_vlm_e0 = optax.global_norm(grad_vlm_e0)
        norm_vlm_e1 = optax.global_norm(grad_vlm_e1)

        diag_metrics["grad_norm_vlm_from_expert0"] = norm_vlm_e0
        diag_metrics["grad_norm_vlm_from_expert1"] = norm_vlm_e1

        dot_e0_e1 = _tree_dot(grad_vlm_e0, grad_vlm_e1)
        dot_total_e0 = _tree_dot(grads_vlm, grad_vlm_e0)
        dot_total_e1 = _tree_dot(grads_vlm, grad_vlm_e1)

        diag_metrics["grad_cos_vlm_e0_e1"] = dot_e0_e1 / (norm_vlm_e0 * norm_vlm_e1 + epsilon)
        diag_metrics["grad_cos_vlm_total_e0"] = dot_total_e0 / (grad_norm_vlm * norm_vlm_e0 + epsilon)
        diag_metrics["grad_cos_vlm_total_e1"] = dot_total_e1 / (grad_norm_vlm * norm_vlm_e1 + epsilon)

        context = model._dual_expert_forward(context_rng, observation, actions, train=True)
        per_expert_loss = jnp.mean(context.expert_mse, axis=(0, 1))
        diag_metrics["loss_mse_expert0"] = per_expert_loss[0]
        diag_metrics["loss_mse_expert1"] = per_expert_loss[1]

        selected_route = context.selected_route.astype(jnp.int32)
        predicted_route = context.predicted_route.astype(jnp.int32)

        expert0_mask = (selected_route == 0)
        expert1_mask = (selected_route == 1)

        valid_elements = jnp.ones(actions.shape, dtype=jnp.float32)
        if observation.action_loss_mask is not None:
            valid_elements *= observation.action_loss_mask[..., None]
        if observation.action_dim_mask is not None:
            dim_mask = observation.action_dim_mask
            if dim_mask.ndim == 2:
                dim_mask = dim_mask[:, None, :]
            valid_elements *= dim_mask
        valid_per_example = jnp.sum(valid_elements, axis=(1, 2))
        # expert_mse uses B*H / total_valid scaling; undo it before applying
        # each route group's own valid-element denominator.
        inverse_loss_scale = jnp.sum(valid_per_example) / (actions.shape[0] * actions.shape[1])

        def _routed_element_mse(expert_index, route_mask):
            route_mask = route_mask.astype(jnp.float32)
            group_sse = jnp.sum(context.expert_mse[..., expert_index] * route_mask[:, None]) * inverse_loss_scale
            group_count = jnp.sum(valid_per_example * route_mask)
            return group_sse / jnp.maximum(group_count, 1.0)

        diag_metrics["loss_mse_expert0_masked"] = _routed_element_mse(0, expert0_mask)
        diag_metrics["loss_mse_expert1_masked"] = _routed_element_mse(1, expert1_mask)

        diag_metrics["route_selected_fraction_0"] = jnp.mean(expert0_mask.astype(jnp.float32))
        diag_metrics["route_selected_fraction_1"] = jnp.mean(expert1_mask.astype(jnp.float32))
        diag_metrics["route_pred_fraction_0"] = jnp.mean((predicted_route == 0).astype(jnp.float32))
        diag_metrics["route_pred_fraction_1"] = jnp.mean((predicted_route == 1).astype(jnp.float32))

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(diag_metrics)
    return new_state, info
